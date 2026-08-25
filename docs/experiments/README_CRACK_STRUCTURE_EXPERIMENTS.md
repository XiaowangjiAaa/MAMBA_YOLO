# Mamba-YOLO 裂缝结构感知扫描与记忆管理实验说明

本文档对应本仓库新增的 `OrientationVSSBlock`、`CrackWriteVSSBlock`、`CrackMemoryVSSBlock` 等模块和配套 YAML。目标不是继续在浅层堆叠门控，而是把两个问题拆开验证：

1. **Scan：信息沿哪个方向传播？** 在保留较多几何细节的 backbone P3 估计局部裂缝方向，并自适应融合水平/垂直扫描结果。
2. **Memory：哪些信息写入和保留？** 在语义更稳定的 backbone P4 估计裂缝概率，优先增强疑似裂缝位置的状态写入；更强版本才同时调节 Delta。

## 1. 为什么选择 P3 和 P4

已有可视化说明浅层 P1/P2 对边缘和细纹响应很充分，但这并不等于已经形成可靠的“裂缝结构语义”。浅层同时会强烈响应接缝、阴影和纹理，因此直接改变其状态更新容易放大背景。P3 仍保留足够空间分辨率，适合估计局部切线；P4 的感受野和语义更稳定，适合预测裂缝概率并管理记忆。P5 分辨率过低，不作为第一轮主要修改位置。

```text
P1/P2：边缘与纹理（观察，不优先门控）
   ↓
P3：局部裂缝几何 → Orientation-aware Scan
   ↓
P4：裂缝语义概率 → Crack-aware Memory Write / Delta
   ↓
P5 + Neck + Segment：保持原版，减少混杂变量
```

## 2. 新增模块

### 2.1 OrientationVSSBlock：方向自适应 H/V Scan

从当前 SSM 输入特征预测二维无向切线编码：

```text
o = tanh(Conv1x1(X))
w = K * softmax(tau * similarity(o, direction_basis))
Y = sum_k w_k * inverse_scan(SSM(scan_k(X)))
```

裂缝方向具有 `theta` 与 `theta + pi` 等价的特点，辅助监督使用 `(cos(2theta), sin(2theta))`，避免同一条裂缝因方向正反产生冲突。当前实现会执行全部方向后做逐像素软融合，因此属于**自适应方向融合**，不是节省计算量的硬路由。

方向头权重和偏置为零初始化，初始 softmax 权重均匀。H/V 版本此时与原 Cross Scan 数值等价，便于加载原模型权重并稳定微调。

### 2.2 DiagonalOrientationVSSBlock：H/V + 两条对角线

把扫描族扩展为水平、垂直、主对角线和副对角线，每族包含正反方向，共 8 条扫描。它对斜裂缝更直接，但计算量更大，而且 K 从 4 变为 8 后扫描参数形状变化，不能完整继承对应 VSS 层的原权重。因此它是第二阶段实验，不应先于 H/V 版本。

### 2.3 CrackWriteVSSBlock：概率门控输入写入

由 P4 特征预测概率图：

```text
g = sigmoid(Conv1x1(X))
B_hat = B * (1 + beta * g)
```

`beta` 被限制在 `(-0.5, 0.5)`，并以 `0` 初始化，因此初始模型与原版严格等价；训练会自行决定增强或抑制概率区域的写入。该版本保持 Delta 不变，只回答“概率引导的状态写入是否有效”。它比之前直接改变 Delta 更保守，也是当前最推荐的 memory 起点。

### 2.4 CrackMemoryVSSBlock：Delta + 写入联合控制

更强版本共享同一概率图，同时进行：

```text
Delta_logits_hat = Delta_logits + alpha * (0.5 - g)
B_hat            = B * (1 + beta * g)
```

`alpha` 和 `beta` 都有界。高概率位置减小 pre-softplus Delta、倾向更长保留，同时增强输入写入；背景反之。它用于检验 Delta 调节在得到概率监督和 B 门控后是否有效。此前 Delta-only 整体下降，因此不要把它作为第一优先级。

### 2.5 CrackStructureVSSBlock

同一层内组合 H/V 方向融合与 B 写入门控。代码已提供，适合后续研究同尺度联合控制；当前主实验仍建议使用“P3 Scan + P4 Write”的跨尺度职责分离，解释更清楚。

## 3. 可选辅助监督

YAML 顶层可设置：

```yaml
guidance_loss_weight: 0.05     # 概率图 BCE + Dice 总权重
guidance_dice_weight: 0.5      # Dice 在概率损失内部的比例
orientation_loss_weight: 0.02  # mask 边界切线监督权重
```

训练损失会从实例 mask 自动生成每张图的 union crack mask；概率监督使用 BCE + Dice，方向监督用 Sobel 梯度得到边界法线，再转换为无向切线。辅助项合并进现有 segmentation loss，训练日志和外部接口仍保持原来的 4 个 loss 分量。

不带 `-sup-` 的配置完全依靠最终检测/分割损失端到端学习；带 `-sup-` 的配置使用上面的弱监督。二者必须成对比较。

## 4. YAML 实验矩阵

所有文件位于 `ultralytics/cfg/models/mamba-yolo/`。

### YOLO11 主矩阵

| 编号 | YAML | 唯一主要变量 | 用途 |
|---|---|---|---|
| B0 | `yolo-mamba-seg-yolo11.yaml` | 无 | 原始基线 |
| S1 | `yolo-mamba-orientation-p3-seg-yolo11.yaml` | P3 H/V Scan | 首个 Scan 消融 |
| S2 | `yolo-mamba-orientation-p3-sup-seg-yolo11.yaml` | S1 + 方向监督 | 验证监督必要性 |
| S3 | `yolo-mamba-orientation-p4-seg-yolo11.yaml` | P4 H/V Scan | P3/P4 位置对照 |
| S4 | `yolo-mamba-orientation-p3p4-seg-yolo11.yaml` | P3+P4 H/V Scan | 多尺度是否互补 |
| S5 | `yolo-mamba-orientation-diagonal-p3-sup-seg-yolo11.yaml` | P3 八向 Scan | 斜向增强，高成本 |
| M1 | `yolo-mamba-crack-write-p4-seg-yolo11.yaml` | P4 B/write | 首个 Memory 消融 |
| M2 | `yolo-mamba-crack-write-p4-sup-seg-yolo11.yaml` | M1 + 概率监督 | 推荐 memory 候选 |
| M3 | `yolo-mamba-crack-memory-p4-sup-seg-yolo11.yaml` | P4 Delta+B | 检查 Delta 是否仍有害 |
| M0 | `yolo-mamba-crack-write-p3-seg-yolo11.yaml` | P3 B/write | P3/P4 memory 层位对照 |
| M0S | `yolo-mamba-crack-write-p3-sup-seg-yolo11.yaml` | P3 B/write + 监督 | P3 gate 监督对照 |
| M4 | `yolo-mamba-crack-write-p3p4-sup-seg-yolo11.yaml` | P3+P4 B/write | 多尺度 memory 对照 |
| C1 | `yolo-mamba-scan-p3-write-p4-seg-yolo11.yaml` | S1+M1 | 无辅助监督组合 |
| C2 | `yolo-mamba-scan-p3-write-p4-sup-seg-yolo11.yaml` | S2+M2 | **当前主推荐组合** |
| C3 | `yolo-mamba-scan-p3-memory-p4-sup-seg-yolo11.yaml` | P3 Scan + P4 Delta+B | 强 memory 对照 |
| J1 | `yolo-mamba-structure-p3p4-sup-seg-yolo11.yaml` | P3/P4 各自 Scan+Write | 同尺度耦合对照 |

### YOLOv8 对照配置

- `yolo-mamba-orientation-p3-seg.yaml`
- `yolo-mamba-crack-write-p4-seg.yaml`
- `yolo-mamba-scan-p3-write-p4-seg.yaml`
- `yolo-mamba-scan-p3-write-p4-sup-seg.yaml`

这些配置保持原 v8 `Segment` head，用来区分改进是否依赖 YOLO11 head。

## 5. 推荐训练顺序

不要一次把所有配置同时视作等价候选。建议按以下阶段推进：

1. 用完全相同的数据划分、随机种子、图像尺寸、epoch 和预训练权重重跑 B0。
2. 第一轮只训练 S1、S2、M1、M2，分别确定 Scan 和 Memory 是否单独有效。
3. 只有当 S 或 M 至少一项有效时，再训练 C1、C2。
4. M3 用于判断 Delta 是否是性能下降源；S3/S4 用于判断层位；S5 最后运行。
5. 最优结构至少运行 3 个随机种子，报告均值和标准差。

单次示例（模型尺寸后缀按项目原有方式选择）：

```bash
yolo segment train model=ultralytics/cfg/models/mamba-yolo/yolo-mamba-scan-p3-write-p4-sup-seg-yolo11.yaml data=YOUR_DATA.yaml imgsz=640 epochs=300 seed=0
```

批量实验时可遍历上述 YAML，但必须给每个 run 唯一名称，并保存完整训练参数。若从 baseline checkpoint 微调，H/V 与 write 版本大部分参数可继承；八向版本对应层会因 K=8 出现形状不匹配，应记录实际加载比例。

## 6. 评价重点

除 Box/Mask mAP50-95 外，裂缝任务至少同时观察：

- Mask Recall：是否减少断裂和漏检。
- Mask AP75：边界和细裂缝定位是否更精确。
- clDice：拓扑连通性是否改善。
- 每张图误检连通域数量：方向传播是否把背景纹理连接成假裂缝。
- Params、GFLOPs、FPS 和显存：尤其是 8-scan 版本。

若 mAP 上升但 clDice 不升，说明模型可能只改善区域重叠，没有真正改善结构连续性。若 S2 明显差于 S1，首先检查 mask 标注边界噪声并把 `orientation_loss_weight` 降到 `0.01`；若 M2 概率图几乎全黑或全白，检查 union mask 生成和类别极不平衡。

## 7. 推荐可视化

优先可视化三类对象，而不是只看普通通道均值热力图：

1. P3 `last_orientation`：用箭头或 HSV 展示预测切线，并与标注裂缝走向对齐。
2. P4 `last_guidance`：检查高概率是否覆盖真实裂缝而非阴影/接缝。
3. P3/P4 block 输出前后差值：判断新模块究竟增强了连续裂缝还是背景纹理。

仓库的 `visualize_model_features.py` 可读取 YAML 和 `.pt/.pth` 并 hook 通用 backbone 层；对新增模块，它还会自动导出 `last_orientation` 的 HSV 无向切线图和 `last_guidance` 概率热力图。若只想看普通特征，可传入 `--no-structure-maps`。

## 8. 代码位置

- 扫描排列、方向融合、Delta/B 调制：`ultralytics/nn/modules/common_utils_mbyolo.py`
- 模块定义与零初始化：`ultralytics/nn/modules/mamba_yolo.py`
- YAML 模块注册和解析：`ultralytics/nn/modules/__init__.py`、`ultralytics/nn/tasks.py`
- mask 概率与方向辅助损失：`ultralytics/utils/loss.py`

## 9. 结果解释原则

本实验体系的关键不是“模块越多越好”，而是建立因果顺序：

```text
S1 vs B0  → 方向扫描本身是否有效
S2 vs S1  → 方向监督是否有效
M1 vs B0  → 写入门控本身是否有效
M2 vs M1  → 概率监督是否有效
C2 vs S2/M2 → 两者是否互补
C3 vs C2  → Delta 调节是否仍是负贡献
S3/S4 vs S1 → P3 是否确实是更合适的 Scan 层位
```

只有这组关系成立，才能把论文主线可靠地表述为：先在 P3 决定沿何种结构传播，再在 P4 决定哪些语义信息值得写入和保留。
