import fs from 'node:fs';
import path from 'node:path';

const root = 'output_dir/crack-seg-9.11';
const labels = {
  K01: ['Structure-aware SSM; Structure-guided State Update; Placement', 'Adaptive path + full update, backbone P2-P5'],
  K02: ['Structure-aware SSM', 'Raster'], K03: ['Structure-aware SSM', 'Cross'],
  K04: ['Structure-aware SSM', 'Serpentine'], K05: ['Structure-aware SSM', 'Diagonal'],
  K06: ['Structure-aware SSM', 'Fixed straight'],
  K10: ['Structure-guided State Update', 'Standard'],
  K11: ['Structure-guided State Update', 'Retention'],
  K12: ['Structure-guided State Update', 'Retention + Transition'],
  K13: ['Structure-guided State Update', 'Retention + Write'],
  K20: ['Placement', 'Backbone P3'], K21: ['Placement', 'Backbone P4'],
  K22: ['Placement', 'Backbone P3+P4'], K23: ['Placement', 'Backbone P2+P3'],
  K24: ['Placement', 'Backbone P4+P5'], K26: ['Placement', 'Neck only'],
  K27: ['Placement', 'Backbone + Neck'],
  K28: ['YOLO family transfer', 'YOLOv9c-Seg baseline'],
  K29: ['YOLO family transfer', 'YOLOv9c-Seg + CPSB'],
  K30: ['YOLO family transfer', 'YOLOv5n-Seg baseline'],
  K31: ['YOLO family transfer', 'YOLOv5n-Seg + CPSB'],
  K32: ['YOLO family transfer', 'YOLOv8n-Seg baseline'],
  K33: ['YOLO family transfer', 'YOLOv8n-Seg + CPSB'],
};
const headers = ['Group','ID','Configuration','Seed','Images','Instances','Box_Precision','Box_Recall','Box_mAP50','Mask_Precision','Mask_Recall','Mask_mAP50','Params_M','GFLOPs_G','Params_exact','Box_mAP75','Box_mAP50_95','Mask_mAP75','Mask_mAP50_95','Mask_mIoU_union','Mask_clDice','Mask_fitness','Fitness','Report_FPS_inference_only','Epochs_recorded','Model_yaml','Source_report'];
const records = [];
for (const dir of fs.readdirSync(root).sort()) {
  const folder = path.join(root,dir);
  if (!fs.statSync(folder).isDirectory()) continue;
  const report = path.join(folder,'segmentation_eval_report_best_map50.txt');
  if (!fs.existsSync(report)) throw Error(`Missing report: ${report}`);
  const text = fs.readFileSync(report,'utf8');
  const args = fs.readFileSync(path.join(folder,'args.yaml'),'utf8');
  const id = dir.split('_')[0];
  if (!labels[id]) throw Error(`Unmapped run ${id}`);
  const capture = (pattern, source=text) => {
    const match = source.match(pattern);
    if (!match) throw Error(`Missing ${pattern} in ${report}`);
    return match[1];
  };
  const metric = key => {
    const line = text.split(/\r?\n/).find(l=>l.trimStart().startsWith(key+' '));
    if (!line) throw Error(`Missing metric ${key}`);
    return line.trim().split(/\s+/).slice(-2).map(v=>v==='-'?null:Number(v));
  };
  const [bp,mp]=metric('Precision'), [br,mr]=metric('Recall');
  const [b50,m50]=metric('mAP@0.5'), [b75,m75]=metric('mAP@0.75'), [b95,m95]=metric('mAP@0.5:0.95');
  const lines=fs.readFileSync(path.join(folder,'results.csv'),'utf8').trim().split(/\r?\n/);
  const row=[...labels[id].slice(0,1),id,labels[id][1],Number(capture(/^seed: (\d+)/m,args)),Number(capture(/Images evaluated:\s+(\d+)/)),Number(capture(/Total instances:\s+(\d+)/)),bp,br,b50,mp,mr,m50,Number(capture(/Parameters:\s+([\d.]+) M/)),Number(capture(/GFLOPs:\s+([\d.]+) G/)),Number(capture(/Parameters:\s+[\d.]+ M \(([\d,]+)\)/).replaceAll(',','')),b75,b95,m75,m95,metric('mIoU (union)')[1],metric('clDice')[1],metric('Mask fitness')[1],Number(capture(/^Fitness\s+([\d.]+)/m)),Number(capture(/FPS:\s+([\d.]+)/)),lines.length-1,capture(/^model: (.+)$/m,args),report.replaceAll('\\','/')];
  if(row.length!==headers.length) throw Error(`Wrong column count ${id}`);
  for(const value of [bp,br,b50,mp,mr,m50,b75,b95,m75,m95]) if(!Number.isFinite(value)||value<0||value>1) throw Error(`Invalid metric ${id}`);
  records.push(row);
}
const escape = v => '"'+String(v??'').replaceAll('"','""')+'"';
const output='docs/experiments/9.11_CPSB_ALL_RESULTS.csv';
fs.writeFileSync(output,'\uFEFF'+[headers,...records].map(r=>r.map(escape).join(',')).join('\r\n')+'\r\n','utf8');
if(fs.readFileSync(output,'utf8').trim().split(/\r?\n/).length!==records.length+1) throw Error('CSV row mismatch');
console.log(`Saved ${records.length} unique runs, ${headers.length} columns: ${output}`);
console.log('ID | Configuration | Box P | Box R | Box AP50 | Mask P | Mask R | Mask AP50 | Params M | GFLOPs G');
for(const r of records.filter(r=>r[1]==='K01'||!r[0].includes('Placement'))) console.log([r[1],r[2],...r.slice(6,12).map(v=>(v*100).toFixed(2)),...r.slice(12,14).map(v=>v.toFixed(2))].join(' | '));
