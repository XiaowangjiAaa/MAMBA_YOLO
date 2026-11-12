#Requires -Version 5.1
<#
.SYNOPSIS
    Mamba-YOLO (crack-seg fork) 一键环境安装脚本（Windows + Conda + NVIDIA GPU）。

.DESCRIPTION
    自动完成以下工作：
      1. 创建 / 复用 Conda 环境（默认名称 mambayolo，Python 3.11）。
      2. 安装 PyTorch 2.3.0 + torchvision 0.18.0 + torchaudio 2.3.0（默认 CUDA 12.1 wheel）。
      3. 安装项目所需的全部 Python 依赖（numpy 1.26、opencv、timm、einops、pycocotools 等）。
      4. 以 editable 模式安装本仓库的 ultralytics 包。
      5. 编译 selective_scan CUDA 扩展（本 fork 只使用 core 模块，默认只编译 core）。
      6. 做一次导入 + CUDA 前向自检，并打印使用说明。

    脚本可以重复运行：环境已存在时默认复用，加 -Recreate 可删除重建。

.PARAMETER EnvName
    Conda 环境名称，默认 mambayolo。

.PARAMETER PythonVersion
    Python 版本，默认 3.11（与 README / torch 2.3.0 匹配）。

.PARAMETER CudaTag
    PyTorch wheel 的 CUDA 版本：cu121（默认，匹配 torch 2.3.0）、cu118 或 cpu。
    注意：cpu 模式无法编译 selective_scan，Mamba 模块不能训练/推理。

.PARAMETER Recreate
    删除同名 Conda 环境后重新创建。

.PARAMETER SkipSelectiveScan
    跳过 selective_scan CUDA 扩展编译（只装 Python 依赖）。

.PARAMETER FullSelectiveScan
    编译 selective_scan 的全部三个模块（core/ndstate/oflex）。
    默认只编译 core，因为本仓库代码实际只调用 selective_scan_cuda_core。

.PARAMETER SkipVerify
    跳过最后的导入 / 前向自检。

.PARAMETER UseChinaMirror
    PyPI 使用清华镜像（PyTorch wheel 仍走官方源）。

.PARAMETER MaxJobs
    并行编译任务数，默认 4（显存/内存较小的机器可设为 2）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_env.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -EnvName mambayolo -Recreate

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -SkipSelectiveScan -UseChinaMirror

.NOTES
    需要预先安装：Miniconda/Anaconda、NVIDIA 驱动、CUDA Toolkit（含 nvcc）、
    Visual Studio 2019/2022 Build Tools（含“使用 C++ 的桌面开发”）。
    全部输出会同时写入脚本目录下的 setup_env_<时间戳>.log。
#>
[CmdletBinding()]
param(
    [string]$EnvName = 'mambayolo',
    [string]$PythonVersion = '3.11',

    [ValidateSet('cu121', 'cu118', 'cpu')]
    [string]$CudaTag = 'cu121',

    [string]$TorchVersion = '2.3.0',
    [string]$TorchVisionVersion = '0.18.0',
    [string]$TorchAudioVersion = '2.3.0',

    [string]$CondaExe = '',
    [string]$CudaHome = '',
    [int]$MaxJobs = 4,

    [switch]$Recreate,
    [switch]$SkipSelectiveScan,
    [switch]$FullSelectiveScan,
    [switch]$SkipVerify,
    [switch]$UseChinaMirror,
    [switch]$KeepTemp
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$RepoRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($RepoRoot)) { $RepoRoot = (Get-Location).Path }
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path

$script:Py = $null
$script:Conda = $null
$script:ScanInstalled = $false
$script:Warnings = New-Object System.Collections.Generic.List[string]

# ---------------------------------------------------------------------------
# 输出 / 日志
# ---------------------------------------------------------------------------
$script:StepNo = 0
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$script:LogFile = Join-Path $RepoRoot "setup_env_$stamp.log"
$script:Transcript = $false
try {
    Start-Transcript -Path $script:LogFile -Force | Out-Null
    $script:Transcript = $true
} catch {
    Write-Host "[WARN] 无法开启日志记录：$($_.Exception.Message)" -ForegroundColor Yellow
}

function Write-Step {
    param([string]$Message)
    $script:StepNo++
    Write-Host ''
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
    Write-Host ("  [{0}] {1}" -f $script:StepNo, $Message) -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
}

function Write-Info { param([string]$Message) Write-Host "  [INFO] $Message" -ForegroundColor Gray }
function Write-Ok   { param([string]$Message) Write-Host "  [ OK ] $Message" -ForegroundColor Green }
function Write-Warn2 {
    param([string]$Message)
    Write-Host "  [WARN] $Message" -ForegroundColor Yellow
    $script:Warnings.Add($Message)
}
function Write-Err2 { param([string]$Message) Write-Host "  [FAIL] $Message" -ForegroundColor Red }

# ---------------------------------------------------------------------------
# 通用命令执行
# ---------------------------------------------------------------------------
function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$ArgList = @(),
        [string]$What = 'command'
    )
    Write-Host ("  > `"{0}`" {1}" -f $Exe, ($ArgList -join ' ')) -ForegroundColor DarkGray
    # 输出统一送到控制台，避免污染调用方的返回值（例如 Install-SelectiveScan 的布尔返回值）。
    & $Exe @ArgList | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "$What 失败（退出码 $LASTEXITCODE）：$Exe $($ArgList -join ' ')"
    }
}

function Invoke-NativeRedirect {
    # 把原生命令的 stdout/stderr 全部写入日志，并返回退出码。
    # Windows PowerShell 5.1 在 $ErrorActionPreference=Stop 时会把重定向的原生 stderr 当成终止错误，
    # 因此这里临时改为 Continue。
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments *> $LogPath
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $old
    }
}

function Invoke-Pip {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$What = 'pip install'
    )
    $all = @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-input') + $Arguments
    Invoke-Checked -Exe $script:Py -ArgList $all -What $What
}

# ---------------------------------------------------------------------------
# Conda / 环境
# ---------------------------------------------------------------------------
function Find-CondaExe {
    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($CondaExe)) { $candidates.Add($CondaExe) }
    $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($cmd) { $candidates.Add($cmd.Source) }
    if ($env:CONDA_EXE) { $candidates.Add($env:CONDA_EXE) }
    $candidates.Add((Join-Path $env:USERPROFILE 'anaconda3\Scripts\conda.exe'))
    $candidates.Add((Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'))
    $candidates.Add((Join-Path $env:USERPROFILE 'miniforge3\Scripts\conda.exe'))
    $candidates.Add((Join-Path $env:LOCALAPPDATA 'miniconda3\Scripts\conda.exe'))
    $candidates.Add((Join-Path $env:LOCALAPPDATA 'anaconda3\Scripts\conda.exe'))
    $candidates.Add('C:\ProgramData\miniconda3\Scripts\conda.exe')
    $candidates.Add('C:\ProgramData\anaconda3\Scripts\conda.exe')
    $candidates.Add('F:\conda\Scripts\conda.exe')

    foreach ($c in $candidates) {
        if (-not [string]::IsNullOrWhiteSpace($c) -and (Test-Path -LiteralPath $c)) {
            return (Resolve-Path -LiteralPath $c).Path
        }
    }
    throw "找不到 conda.exe。请安装 Miniconda/Anaconda，或用 -CondaExe 指定完整路径。"
}

function Get-CondaEnvPaths {
    $json = & $script:Conda env list --json
    if ($LASTEXITCODE -ne 0) { throw 'conda env list 执行失败。' }
    $obj = ($json | Out-String) | ConvertFrom-Json
    return @($obj.envs)
}

function Resolve-EnvPython {
    param([string]$EnvPath)
    $py = Join-Path $EnvPath 'python.exe'
    if (-not (Test-Path -LiteralPath $py)) {
        throw "环境目录中找不到 python.exe：$EnvPath"
    }
    return $py
}

function Ensure-CondaEnvironment {
    $envs = Get-CondaEnvPaths
    $existing = @($envs | Where-Object {
            $_ -and ((Split-Path -Leaf $_) -ieq $EnvName)
        })

    if ($existing.Count -gt 0 -and $Recreate) {
        Write-Info "检测到已有环境 $EnvName，按 -Recreate 删除重建 ..."
        & $script:Conda env remove -y -n $EnvName
        if ($LASTEXITCODE -ne 0) { throw "删除 Conda 环境 $EnvName 失败（环境可能正在使用）。" }
        $existing = @()
    }

    if ($existing.Count -gt 0) {
        $envPath = $existing[0]
        Write-Ok "复用已有 Conda 环境：$envPath"
        $script:Py = Resolve-EnvPython -EnvPath $envPath
    } else {
        Write-Info "创建 Conda 环境 $EnvName (python=$PythonVersion) ..."
        & $script:Conda create -y -n $EnvName "python=$PythonVersion"
        if ($LASTEXITCODE -ne 0) {
            throw @"
创建 Conda 环境失败。
如果提示 Terms of Service 未接受，请先运行：
    conda tos accept --override-channels --channel defaults
然后重新执行本脚本。
"@
        }
        $envs = Get-CondaEnvPaths
        $envPath = @($envs | Where-Object { $_ -and ((Split-Path -Leaf $_) -ieq $EnvName) }) | Select-Object -First 1
        if (-not $envPath) { throw "Conda 环境创建后仍未找到：$EnvName" }
        $script:Py = Resolve-EnvPython -EnvPath $envPath
    }

    $actual = (& $script:Py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    Write-Ok "Python 解释器：$script:Py（Python $actual）"
    if ($actual -ne $PythonVersion) {
        Write-Warn2 "环境 Python 版本为 $actual，与请求的 $PythonVersion 不一致；如需精确匹配请加 -Recreate。"
    }
}

# ---------------------------------------------------------------------------
# PyTorch / Python 依赖
# ---------------------------------------------------------------------------
function Get-PipMirrorArgs {
    if ($UseChinaMirror) {
        return @('-i', 'https://pypi.tuna.tsinghua.edu.cn/simple')
    }
    return @()
}

function Install-PipTooling {
    Write-Info '升级 pip / setuptools / wheel ...'
    Invoke-Pip -Arguments @('--upgrade', 'pip', 'setuptools', 'wheel') -What '升级 pip 工具链'
    Write-Ok 'pip 工具链已就绪。'
}

function Install-TorchStack {
    if ($CudaTag -eq 'cpu') {
        $index = 'https://download.pytorch.org/whl/cpu'
    } else {
        $index = "https://download.pytorch.org/whl/$CudaTag"
    }

    $specs = @(
        "torch==$TorchVersion",
        "torchvision==$TorchVisionVersion",
        "torchaudio==$TorchAudioVersion",
        # 固定 numpy 1.x，避免 PyTorch 索引把 numpy 2.x 作为依赖装上（torch 2.3 与 numpy 2 不兼容）。
        "numpy==1.26.4"
    )
    Write-Info "从 $index 安装 PyTorch $TorchVersion ..."
    Invoke-Pip -Arguments (@('--index-url', $index) + $specs) -What '安装 PyTorch'

    $check = & $script:Py -c "import torch, torchvision; print(torch.__version__, torch.version.cuda, torchvision.__version__, torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch 安装后无法导入。' }
    Write-Ok "PyTorch：$check"
}

function Install-PythonPackages {
    # 版本锁定在 torch 2.3.0 / Python 3.11 验证过的组合上，避免 numpy 2.x 等兼容性问题。
    $packages = @(
        'numpy==1.26.4',
        'opencv-python==4.10.0.84',
        'pillow==10.4.0',
        'PyYAML==6.0.2',
        'requests==2.32.3',
        'scipy==1.13.1',
        'tqdm==4.66.5',
        'psutil==6.0.0',
        'py-cpuinfo==9.0.0',
        'pandas==2.2.3',
        'seaborn==0.13.2',
        'matplotlib==3.9.2',
        'ultralytics-thop==2.0.0',
        'timm==1.0.9',
        'einops==0.8.0',
        'packaging>=23.2',
        'ninja>=1.11',
        'pycocotools==2.0.8'
    )
    Write-Info '安装项目 Python 依赖 ...'
    Invoke-Pip -Arguments (@(Get-PipMirrorArgs) + $packages) -What '安装 Python 依赖'

    # tensorboard 只用于训练曲线，失败不影响主流程。
    try {
        Invoke-Pip -Arguments (@(Get-PipMirrorArgs) + @('tensorboard==2.17.1')) -What '安装 tensorboard'
        Write-Ok 'tensorboard 已安装。'
    } catch {
        Write-Warn2 "tensorboard 安装失败（不影响训练）：$($_.Exception.Message)"
    }
    Write-Ok 'Python 依赖安装完成。'
}

function Install-UltralyticsPackage {
    Write-Info '安装本仓库 ultralytics ...'
    $editableOk = $true
    $editableLog = Join-Path ([System.IO.Path]::GetTempPath()) ("mbyolo_editable_" + [guid]::NewGuid().ToString('N') + '.log')
    # --no-deps：依赖已在上一步精确锁定，避免 pip 解析器改动 torch/numpy 版本。
    $editableArgs = @(
        '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input',
        '--no-deps', '--no-build-isolation', '-e', $RepoRoot
    )
    $editableCode = Invoke-NativeRedirect -Exe $script:Py -Arguments $editableArgs -LogPath $editableLog
    if ($editableCode -ne 0) {
        $editableOk = $false
        Write-Warn2 "editable 安装不可用：本项目版本号 8.2.29-mamba-yolo11 不符合 PEP 440（新版 setuptools 会拒绝），已自动改用 .pth 方式。"
        Write-Info '改用 .pth 文件把仓库目录加入 site-packages（效果等同于 editable 安装）。'
    }
    if ($KeepTemp) {
        Write-Info "editable 安装日志：$editableLog"
    } elseif (Test-Path -LiteralPath $editableLog) {
        Remove-Item -LiteralPath $editableLog -Force -ErrorAction SilentlyContinue
    }

    if (-not $editableOk) {
        $purelib = (& $script:Py -c "import sysconfig; print(sysconfig.get_paths()['purelib'])").Trim()
        if (-not $purelib -or -not (Test-Path -LiteralPath $purelib)) {
            throw "无法定位 site-packages：$purelib"
        }
        $pth = Join-Path $purelib 'mamba_yolo_repo.pth'
        Set-Content -LiteralPath $pth -Value $RepoRoot -Encoding ASCII
        Write-Ok "已写入路径文件：$pth"
    }

    # 注意：此处不能 import ultralytics —— Mamba 模块依赖尚未编译的 selective_scan。
    # 真正的导入检查放在最后的环境自检里（此时 selective_scan 已就绪）。
    Write-Ok "ultralytics 已注册：$RepoRoot"
}

# ---------------------------------------------------------------------------
# CUDA / MSVC 编译环境
# ---------------------------------------------------------------------------
function Resolve-CudaHome {
    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($CudaHome)) { $candidates.Add($CudaHome) }
    if ($env:CUDA_HOME) { $candidates.Add($env:CUDA_HOME) }
    if ($env:CUDA_PATH) { $candidates.Add($env:CUDA_PATH) }

    $root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $root) {
        $dirs = Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
            Sort-Object -Property @{ Expression = { try { [version]($_.Name -replace '^v', '') } catch { [version]'0.0' } } } -Descending
        foreach ($d in $dirs) { $candidates.Add($d.FullName) }
    }

    $nvccCmd = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($nvccCmd) {
        $candidates.Add((Split-Path -Parent (Split-Path -Parent $nvccCmd.Source)))
    }

    foreach ($c in $candidates) {
        if (-not [string]::IsNullOrWhiteSpace($c) -and (Test-Path -LiteralPath (Join-Path $c 'bin\nvcc.exe'))) {
            return (Resolve-Path -LiteralPath $c).Path
        }
    }
    return $null
}

function Enable-MsvcEnvironment {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $instances = @()
    if (Test-Path -LiteralPath $vswhere) {
        $json = & $vswhere -all -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json
        foreach ($item in @($json | Out-String | ConvertFrom-Json)) {
            $major = 0
            if ($item.installationVersion -match '^(\d+)\.') { $major = [int]$Matches[1] }
            $instances += [pscustomobject]@{ Path = $item.installationPath; Major = $major }
        }
    }

    if ($instances.Count -eq 0) {
        $fallback = @(
            'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools',
            'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools',
            'C:\Program Files (x86)\Microsoft Visual Studio\2022\Community',
            'C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional',
            'C:\Program Files\Microsoft Visual Studio\2022\Community'
        )
        foreach ($p in $fallback) {
            if (Test-Path -LiteralPath $p) {
                $instances += [pscustomobject]@{ Path = $p; Major = 17 }
            }
        }
    }

    # 优先 VS 2019/2022（CUDA 官方支持），其次其它版本（需 -allow-unsupported-compiler）。
    $ordered = $instances | Sort-Object -Property @{
        Expression = {
            switch ($_.Major) {
                17 { 0 }
                16 { 1 }
                18 { 3 }
                default { 2 }
            }
        }
    }, @{ Expression = { $_.Major }; Descending = $true }

    foreach ($inst in $ordered) {
        $vcvars = Join-Path $inst.Path 'VC\Auxiliary\Build\vcvars64.bat'
        if (-not (Test-Path -LiteralPath $vcvars)) { continue }

        $probe = & cmd.exe /c "call `"$vcvars`" >nul 2>&1 && where cl >nul 2>&1 && echo MSVC_OK"
        if (-not ($probe -match 'MSVC_OK')) { continue }

        $envLines = & cmd.exe /c "call `"$vcvars`" >nul 2>&1 && set"
        foreach ($line in $envLines) {
            $eq = $line.IndexOf('=')
            if ($eq -gt 0) {
                $name = $line.Substring(0, $eq)
                $value = $line.Substring($eq + 1)
                try { Set-Item -Path "Env:$name" -Value $value } catch { }
            }
        }
        if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { continue }

        Write-Ok "MSVC 工具链：$($inst.Path)（VS $($inst.Major).x）"
        return $inst.Major
    }

    return $null
}

# ---------------------------------------------------------------------------
# selective_scan CUDA 扩展
# ---------------------------------------------------------------------------
function New-PatchedSource {
    param([string]$SourceDir, [string[]]$Modes)
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("mbyolo_selective_scan_" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    Copy-Item -LiteralPath (Join-Path $SourceDir 'csrc') -Destination (Join-Path $tmp 'csrc') -Recurse
    Copy-Item -LiteralPath (Join-Path $SourceDir 'setup.py') -Destination (Join-Path $tmp 'setup.py')

    $setupPath = Join-Path $tmp 'setup.py'
    $text = [System.IO.File]::ReadAllText($setupPath)
    $modesLiteral = 'MODES = ["' + ($Modes -join '", "') + '"]'
    $patched = [regex]::Replace($text, 'MODES\s*=\s*\[[^\]]*\]', $modesLiteral)
    if ($patched -eq $text) {
        throw "无法在 setup.py 中定位 MODES 定义：$setupPath"
    }
    [System.IO.File]::WriteAllText($setupPath, $patched, (New-Object -TypeName System.Text.UTF8Encoding -ArgumentList $false))

    # 新版 MSVC (VS 2026 / 19.50+) 不再接受 lambda 内普通 constexpr 变量作为模板实参，
    # 会报 C2975；改为 static constexpr 后 VS2019/2022/2026 均可编译。
    $switchPath = Join-Path $tmp 'csrc\selective_scan\static_switch.h'
    $switchText = [System.IO.File]::ReadAllText($switchPath)
    $switchPatched = $switchText -replace 'constexpr bool CONST_NAME', 'static constexpr bool CONST_NAME'
    if ($switchPatched -eq $switchText) {
        throw "无法在 static_switch.h 中定位 BOOL_SWITCH 定义：$switchPath"
    }
    [System.IO.File]::WriteAllText($switchPath, $switchPatched, (New-Object -TypeName System.Text.UTF8Encoding -ArgumentList $false))

    return $tmp
}

function Build-SelectiveScanModule {
    param(
        [string]$SourceDir,
        [string[]]$Modes,
        [string]$Label
    )
    $tmp = New-PatchedSource -SourceDir $SourceDir -Modes $Modes
    $log = Join-Path ([System.IO.Path]::GetTempPath()) ("mbyolo_selective_scan_build_" + [guid]::NewGuid().ToString('N') + '.log')
    Write-Info "编译 $Label（约 5-10 分钟，详细日志：$log）..."
    try {
        $pipArgs = @(
            '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input',
            '--no-build-isolation', '--no-deps', $tmp
        )
        $buildCode = Invoke-NativeRedirect -Exe $script:Py -Arguments $pipArgs -LogPath $log
        if ($buildCode -ne 0) {
            Write-Err2 "$Label 编译失败，日志最后 80 行："
            Get-Content -LiteralPath $log -Tail 80 -ErrorAction SilentlyContinue |
                ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
            throw "$Label 编译失败（退出码 $buildCode），完整日志：$log"
        }
        if (-not $KeepTemp) {
            Remove-Item -LiteralPath $log -Force -ErrorAction SilentlyContinue
        }
    } finally {
        if (-not $KeepTemp -and (Test-Path -LiteralPath $tmp)) {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Install-SelectiveScan {
    # 先检查是否已经能导入；能导入就不重复编译（重跑脚本时节省 5-10 分钟）。
    if (-not $FullSelectiveScan) {
        $probeLog = Join-Path ([System.IO.Path]::GetTempPath()) ("mbyolo_scan_probe_" + [guid]::NewGuid().ToString('N') + '.log')
        $probeCode = Invoke-NativeRedirect -Exe $script:Py -Arguments @('-c', 'import torch, selective_scan_cuda_core') -LogPath $probeLog
        Remove-Item -LiteralPath $probeLog -Force -ErrorAction SilentlyContinue
        if ($probeCode -eq 0) {
            Write-Ok '检测到可用的 selective_scan_cuda_core，跳过编译。'
            return $true
        }
    }

    if ($SkipSelectiveScan) {
        Write-Warn2 '已按 -SkipSelectiveScan 跳过 selective_scan 编译；Mamba 模块将无法使用。'
        return $false
    }
    if ($CudaTag -eq 'cpu') {
        Write-Warn2 'CPU 模式不编译 selective_scan；Mamba 模块将无法使用。'
        return $false
    }

    $sourceDir = Join-Path $RepoRoot 'selective_scan'
    if (-not (Test-Path -LiteralPath (Join-Path $sourceDir 'setup.py'))) {
        throw "找不到 selective_scan 源码目录：$sourceDir"
    }

    $cuda = Resolve-CudaHome
    if (-not $cuda) {
        Write-Warn2 '未找到 CUDA Toolkit（nvcc）。请安装 CUDA 12.x 并重新运行，或加 -SkipSelectiveScan。'
        return $false
    }
    $env:CUDA_HOME = $cuda
    Write-Info "CUDA_HOME = $cuda"
    $nvccVersion = (& (Join-Path $cuda 'bin\nvcc.exe') --version | Select-String -Pattern 'release' | Select-Object -First 1)
    Write-Info "nvcc：$nvccVersion"

    $msvcMajor = Enable-MsvcEnvironment
    if (-not $msvcMajor) {
        Write-Warn2 '未找到可用的 MSVC (cl.exe)。请安装 Visual Studio Build Tools 的“使用 C++ 的桌面开发”工作负载。'
        return $false
    }

    $env:DISTUTILS_USE_SDK = '1'
    $env:MSSdk = '1'
    $env:VSLANG = '1033'
    $env:MAX_JOBS = [string]([Math]::Max(1, $MaxJobs))
    $env:PYTHONUTF8 = '1'

    # CUDA 12.6 的 host_config.h 目前只声明支持到 VS 2022；
    # 新版 MSVC 需要 -allow-unsupported-compiler，且部分 MSVC 头文件不再默认提供 M_LOG2E。
    $nvccFlags = @('-D_USE_MATH_DEFINES')
    if ($msvcMajor -ge 18) {
        $nvccFlags = @('-allow-unsupported-compiler') + $nvccFlags
    }
    $env:NVCC_PREPEND_FLAGS = ($nvccFlags -join ' ')
    Write-Info "NVCC_PREPEND_FLAGS = $($env:NVCC_PREPEND_FLAGS)"

    $installed = $false

    if ($FullSelectiveScan) {
        try {
            Build-SelectiveScanModule -SourceDir $sourceDir -Modes @('core', 'ndstate', 'oflex') -Label 'selective_scan 全部模块（core + ndstate + oflex）'
            $installed = $true
        } catch {
            Write-Warn2 "全部模块编译失败，回退到只编译 core 模块。原因：$($_.Exception.Message)"
        }
    }

    if (-not $installed) {
        Build-SelectiveScanModule -SourceDir $sourceDir -Modes @('core') -Label 'selective_scan core 模块（本仓库实际只用它）'
        $installed = $true
    }

    if (-not $installed) { return $false }

    # 必须先 import torch：扩展依赖 torch 的 DLL，未被 import 时 Windows 会报“找不到指定的模块”。
    & $script:Py -c "import torch; import selective_scan_cuda_core; print('selective_scan_cuda_core =', selective_scan_cuda_core.__file__)" | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Warn2 'selective_scan 编译完成，但导入失败。'
        return $false
    }
    Write-Ok 'selective_scan CUDA 扩展可用。'
    return $true
}

# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
function Invoke-Verification {
    if ($SkipVerify) {
        Write-Warn2 '已按 -SkipVerify 跳过自检。'
        return
    }
    if (-not $script:ScanInstalled) {
        Write-Warn2 'selective_scan 未安装，跳过 ultralytics / Mamba 前向自检。'
        return
    }

    $verifyScript = Join-Path ([System.IO.Path]::GetTempPath()) ("mbyolo_verify_" + [guid]::NewGuid().ToString('N') + '.py')
    $code = @'
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])

import torch
import torchvision

print(f"[verify] torch {torch.__version__} (CUDA {torch.version.cuda}), cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[verify] GPU: {torch.cuda.get_device_name(0)}")

import cv2
import einops
import matplotlib
import numpy
import pandas
import scipy
import seaborn
import timm
import yaml

print(f"[verify] numpy {numpy.__version__}, opencv {cv2.__version__}, scipy {scipy.__version__}, timm {timm.__version__}")

import selective_scan_cuda_core

print(f"[verify] selective_scan_cuda_core: {selective_scan_cuda_core.__file__}")

import ultralytics

print(f"[verify] ultralytics {ultralytics.__version__}")

if not torch.cuda.is_available():
    print("[verify] 无可用 CUDA，跳过 Mamba 前向测试。")
    sys.exit(0)

from ultralytics import YOLO

cfg = repo / "ultralytics" / "cfg" / "models" / "mamba-yolo" / "Mamba-YOLO-T.yaml"
model = YOLO(str(cfg))
model.model.eval().to("cuda:0")
x = torch.randn(1, 3, 128, 128, device="cuda:0")
with torch.no_grad():
    model.model(x)
print("[verify] Mamba-YOLO 前向测试通过。")
'@
    Set-Content -LiteralPath $verifyScript -Value $code -Encoding UTF8
    try {
        Invoke-Checked -Exe $script:Py -ArgList @($verifyScript, $RepoRoot) -What '环境自检'
    } finally {
        if (-not $KeepTemp -and (Test-Path -LiteralPath $verifyScript)) {
            Remove-Item -LiteralPath $verifyScript -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Ok '自检通过。'
}

# ---------------------------------------------------------------------------
# 总结
# ---------------------------------------------------------------------------
function Show-Summary {
    param([int]$ExitCode)
    Write-Host ''
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
    if ($ExitCode -eq 0) {
        Write-Host '  环境安装完成' -ForegroundColor Green
    } else {
        Write-Host '  环境安装未完全成功，请检查上面的 [FAIL] / [WARN]' -ForegroundColor Yellow
    }
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
    Write-Host "  Conda 环境 : $EnvName"
    Write-Host "  Python     : $script:Py"
    Write-Host "  日志       : $script:LogFile"
    Write-Host ''
    Write-Host '  激活环境：' -ForegroundColor White
    Write-Host "      conda activate $EnvName" -ForegroundColor Cyan
    Write-Host ''
    Write-Host '  验证：' -ForegroundColor White
    Write-Host '      python -c "import torch; print(torch.cuda.is_available())"' -ForegroundColor Cyan
    Write-Host ''
    Write-Host '  开始训练（示例，数据路径按需修改）：' -ForegroundColor White
    Write-Host '      python mbyolo_train.py --task train --data ..\crack-seg\crack-seg.yaml --config ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml --amp --device 0 --project ./output_dir/crack-seg --name my_run' -ForegroundColor Cyan
    Write-Host ''

    if ($script:Warnings.Count -gt 0) {
        Write-Host '  警告：' -ForegroundColor Yellow
        foreach ($w in $script:Warnings) { Write-Host "      - $w" -ForegroundColor Yellow }
        Write-Host ''
    }
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
$exitCode = 0
try {
    Write-Host ''
    Write-Host '  Mamba-YOLO 一键环境安装' -ForegroundColor Cyan
    Write-Host "  仓库：$RepoRoot" -ForegroundColor DarkGray
    Write-Host "  环境：$EnvName | Python $PythonVersion | PyTorch $TorchVersion | $CudaTag" -ForegroundColor DarkGray

    Write-Step '检查 Conda'
    $script:Conda = Find-CondaExe
    $condaVersion = (& $script:Conda --version | Out-String).Trim()
    Write-Ok "$condaVersion（$script:Conda）"

    Write-Step '创建 / 复用 Conda 环境'
    Ensure-CondaEnvironment

    Write-Step '安装 pip 工具链'
    Install-PipTooling

    Write-Step '安装 PyTorch'
    Install-TorchStack

    Write-Step '安装 Python 依赖'
    Install-PythonPackages

    Write-Step '安装本仓库 ultralytics'
    Install-UltralyticsPackage

    Write-Step '编译 selective_scan CUDA 扩展'
    $script:ScanInstalled = Install-SelectiveScan

    Write-Step '环境自检'
    Invoke-Verification

    if (-not $script:ScanInstalled) {
        $exitCode = 1
    }
} catch {
    Write-Err2 $_.Exception.Message
    if ($_.ScriptStackTrace) { Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray }
    $exitCode = 1
} finally {
    Show-Summary -ExitCode $exitCode
    if ($script:Transcript) {
        try { Stop-Transcript | Out-Null } catch { }
    }
}

exit $exitCode
