"""Human-readable final delivery from completed evidence, no GPU or model mutations."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'));BASE=ROOT/'experiments/defect_algorithm_steps_20261112_v2'
from joint_state_v1 import atomic_json,sha
def read(p):return json.loads(p.read_text(encoding='utf8'))
def main():
    from analyze_defect_mechanisms_20261112 import analyze
    mechanisms=analyze()['modes']
    final=read(BASE/'final_review/results.json');assert final['phase']=='complete';base=final['baseline']['fixed'];bt=base['total'];oof=read(BASE/'all_mode_OOF_calibration/results.json');train=read(BASE/'train/results.json');ap=read(BASE/'deployment_ap.json');phase=read(BASE/'phase1/results.json');parity=read(BASE/'mode_deployment_validation/results.json');candidate=final['review_candidate']
    runtime=read(BASE/'runtime_fixed_verification/results.json');assert runtime['all_counts_and_GT_equal'],'Runtime flag discrepancy needs explicit review before delivery'
    for r in train['arms'].values():assert sha(r['checkpoint'])==r['checkpoint_sha256']
    init=read(BASE/'oof_v2/protocol.json')['init_sha256']
    for fi in range(3):
        run=ROOT/f'runs/defect_oof3_fold{fi}_e40_20261112_v1';receipt=read(BASE/f'oof_v2/fold{fi}/predictions.json')
        assert sha(run/'weights/last.pt')==receipt['checkpoint_sha256']
        preflight=read(ROOT/f'preflight/defect_oof3_fold{fi}_e40_20261112_v1.json');assert preflight['input_audit']['weights_sha256']==init
    atomic_json(BASE/'delivery_hash_audit.json',{'six_training_checkpoints_verified':True,'three_OOF_final_checkpoints_verified':True,'OOF_same_initial_COCO_sha256':init})
    names={'o2o_nms':'原单头＋NMS','o2m_nms':'one2many＋NMS','heads_wbf_nms':'双头WBF＋NMS','o2o_wbf':'单头WBF','flip_nms':'640翻转＋NMS','flip_wbf':'640翻转WBF','G0_control':'G0 继续训练','G1_shield':'G1 仅负样本屏蔽','G2_sampling':'G2 仅2.5倍采样','G3_scale':'G3 仅尺度扰动','G4_beta3':'G4 仅TAL beta3','G5_combined':'G5 四项组合'}
    def row(name,m,s,threshold,aps):
        t=m['total'];return f"| {name} | {threshold} | {100*t['P']:.2f} | {100*t['R']:.2f} | {t['TP']}/{t['FP']}/{t['FN']} | {100*m['class']['0']['R']:.2f} | {100*m['class']['1']['R']:.2f} | {m['large']['TP']}/22 | {100*aps['mAP50_95']:.2f} | {s['serial_fps']:.2f} | {s['p95_ms']:.2f} |"
    header=['| 方案 | 类别阈值 | P % | R % | TP/FP/FN | class0 R % | class1 R % | 大框TP | 自定义AP50–95 % | 双任务FPS | P95 ms |','|---|---|---|---|---|---|---|---|---|---|---|']
    lines=['# 缺陷零改标算法实验：最终交付','', '本轮已完成用户指定的推理对照、训练集Ignore Shield、真正的分组OOF类别阈值，以及受控微调与组合实验。原训练/开发GT、原缺陷强权重与全局Base＋Small保持不变。','',
      '## 结论','']
    if candidate:
        t=candidate['fixed']['total'];s=candidate['sustained_speed'];lines += [f"发现值得进一步复核的候选 **{candidate['name']}**：开发集 P **{100*t['P']:.2f}%**、R **{100*t['R']:.2f}%**，TP/FP/FN={t['TP']}/{t['FP']}/{t['FN']}；相对原NMS基线分别变化 **{100*(t['P']-bt['P']):+.2f}**、**{100*(t['R']-bt['R']):+.2f}** 个百分点。",'',f"持续双任务实测 **{s['serial_fps']:.2f} FPS**，均时 {s['mean_ms']:.2f} ms，P95 {s['p95_ms']:.2f} ms。平均20FPS目标{'达到' if candidate['speed_target_met'] else '未达到'}。它是开发集筛选出的候选，尚不是独立测试或多种子重复验证的最终结论；默认部署入口和原权重均未被覆盖。"]
        lines += ['',f"逐GT比较：恢复原漏检 {len(candidate['fixed']['recovered_FN'])} 个，同时新增漏检 {len(candidate['fixed']['new_FN_vs_baseline'])} 个，净减少漏检 {bt['FN']-t['FN']} 个；不是所有旧TP都得到保留。class0召回仍为 {100*candidate['fixed']['class']['0']['R']:.2f}%，没有突破该组召回瓶颈。",f"统一真实部署路径重新完整推理后，自定义AP50–95为基线 {100*final['baseline_exact_AP']['mAP50_95']:.2f}% → 候选 {100*candidate['exact_deployment_AP']['mAP50_95']:.2f}%。表1初始缓存的基线AP来自batch2采集，低分排序有轻微数值差异；最终判定使用这里的实际batch1基线与batch2候选完整重推结果。"]
    else:lines += ['本轮没有候选同时通过预先声明的复核条件：完整开发集P/R均不低于原NMS基线、至少一项提高、大框召回不下降、AP不下降且短程双任务≥20FPS。因此保留原强模型＋显式NMS作为保底，不把单项提分或速度达标包装成准确率突破。']
    lines+=['','## 1. 无训练推理对照（完整113图）','',*header]
    for name,r in phase['modes'].items():lines.append(row(names[name],parity['modes'][name],r['speed'],'0.25 / 0.25',ap[name]))
    lines+=['','各模式均补做真实部署batch形状的完整推理复核。双视图TTA仅用于缺陷分支，成熟度分割仍是单视图。WBF使用预先固定的重叠聚类与均分支持度配方；其低分成员会影响最终分数，这一对照失败不代表所有融合配方均无效。','',
      '## 2. 真正OOF选阈值后的完整开发集结果','',
      '三折分别从COCO初始化、固定40轮最后权重；同秒组不跨折。类别阈值仅用596张训练图的折外预测选择，冻结后在原强权重上评估113张开发图。OOF模型仅用来校准，不替换部署强权重。不同类别阈值的数值不能跨模型或分支直接迁移。','',*header]
    for name,r in oof['modes'].items():lines.append(row(names[name],r['fixed'],r['speed'],' / '.join(map(str,r['thresholds'])),ap[name]))
    lines+=['','OOF共同约束：各类FP不超过OOF原单头0.25基线且P不下降，优先最大TP，再最小FP；无可行阈值的方案会标记为拒绝。该约束不保证跨日期开发集的FP也不增加。物理果实ID未知，同秒分组不是严格同果隔离的证明。','',
      '## 3. 六组训练对照','',*header]
    for name,r in train['arms'].items():lines.append(row(names[name],r['fixed'],r['speed'],'0.25 / 0.25',ap[name]))
    lines+=['','全部同强权重起点、同3轮/114步、同FP32/640/AdamW lr=1e-5。G0是本轮显式仿射增强流水线的继续训练对照；G1–G4分别改变一项，G5组合四项。每轮固定596次抽样；过采样不会增加总步数。TAL与WIoU/Inner-IoU按建议中的“或”选择TAL beta6→3路线，未再叠加回归损失。','',
      '| 组 | 相对G0 P变化(pp) | 相对G0 R变化(pp) | 相对原NMS新增漏检 | 恢复原漏检 |','|---|---|---|---|---|']
    control=train['arms']['G0_control']['fixed']['total']
    for name,r in train['arms'].items():
        t=r['fixed']['total'];lines.append(f"| {names[name]} | {100*(t['P']-control['P']):+.2f} | {100*(t['R']-control['R']):+.2f} | {len(r['new_FN_vs_baseline'])} | {len(r['recovered_FN'])} |")
    lines+=['','| 组 | 恢复原37个低分漏检 | 无近邻标注FP | 定位偏移FP | 错类FP | 重复FP |','|---|---|---|---|---|---|']
    for name in ['baseline',*train['arms']]:
        m=mechanisms[name];f=m['FP_categories'];lines.append(f"| {names.get(name,name)} | {m['original_low37_recovered']}/37 | {f['no_near_annotated_GT']} | {f['localization_0.1_to_0.5']} | {f['wrong_class']} | {f['duplicate']} |")
    lines+=['','Ignore Shield来自原强模型训练内预测的102个候选（82/596图；class0=28、class1=74），不是确认的漏标。它只移除候选区背景锚点的分类惩罚，不新增伪正例；GT内或已分配的正锚点受保护。13项CPU损失/几何检查、7项阈值前缀/AP检查，以及每组2步GPU小批量检查通过。','',
      '不能把“框正确但分数低”直接推断成“漏标造成”。若G1没有优于G0，说明本轮该机制未提供支持；不能再用原假说解释为已证实的因果。TAL目标包含归一化，不是简单把IoU^6当最终分类目标。','',
      '## 4. 持续速度与权重','',
      '| 模式 | 次数 | 均时ms | P95 ms | 平均FPS | 最慢40帧块FPS | 超50ms帧 |','|---|---|---|---|---|---|---|']
    for k,s in final['speed'].items():lines.append(f"| {k} | {s['n']} | {s['mean_ms']:.2f} | {s['p95_ms']:.2f} | {s['serial_fps']:.2f} | {s['worst_40_frame_cycle_fps']:.2f} | {s['slow_frames_over50ms']} |")
    lines+=['','持续测速使用40张图重复15轮/模式，基线与候选交替执行，每个模式600次，包含成熟度分割和缺陷检测从内存BGR到CPU Results的全部路径。排除磁盘读取、相机、深度、机器人控制；重复图不是600张独立样本；平均FPS不保证每帧≤50ms。','训练组短程测速沿用训练进程的确定性设置，不能把其与初始推理组之间的全部延迟差异归因于权重。最终持续测速统一到同一进程设置；并在该设置下重新完整推理113图，确认基线和候选固定工作点的计数及逐GT漏检均一致。','',
      '原强模型：`experiments/strong_defect_composite_20261112_v1/composite_frozen.pt`，SHA256 `f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6`。六组权重及SHA分别记录在 `experiments/defect_algorithm_steps_20261112_v2/train/<组>/result.json`；全部仅在F盘，不上传GitHub。','']
    if candidate:
        cp=Path(candidate['checkpoint']);config={'checkpoint':str(cp),'sha256':sha(cp),'mode':candidate['mode'],'thresholds':candidate['thresholds'],'status':'review candidate, opt-in only','source_result':str(BASE/'final_review/results.json')};atomic_json(ROOT/'configs/defect_review_candidate_20261112.json',config)
        lines += [f"复核候选权重：`{cp}`，SHA256 `{config['sha256']}`；推理模式 `{candidate['mode']}`，阈值 `{candidate['thresholds']}`。参数另存 `configs/defect_review_candidate_20261112.json`。",'']
    lines+=['可选推理入口 `apple_research_v2/predict_defect_algorithm_20261112.py` 支持 `--image`、`--output`、`--mode`、两类 `--thresholds`、`--checkpoint` 和 `--checkpoint-sha256`。默认仍是原强模型＋NMS0.7、两类阈值0.25，使用候选需显式传参。输出独立成熟度掩膜、缺陷框和叠加图；不冒充缺陷分割、果实品质关联或三维定位。','',
      '## 5. 结论适用范围与文件索引','',
      '- P/R按框IoU≥0.5固定工作点计算；表中AP是显式NMS/声明融合后的自定义101点AP，不能与Ultralytics原生AP或原全局苹果94.43%混比。原生评估结果仍保留在各组official_val中。',
      '- 所有结果都是反复使用过的开发集上的探索性对照；还没有独立测试、多种子复现或果园域验证。公开室内果图的标注歧义不会因算法运行完毕而自动消失。',
      '- 任何负结果只否定本次固定预算/配方，不等于证明该方法在所有训练条件下无效。本轮没有无限追加训练或搜索开发集阈值。',
      '- 完整逐GT新增漏检与恢复清单：`mode_deployment_validation/results.json`、`all_mode_OOF_calibration/results.json`、`train/<组>/result.json`。',
      '- 完整来源、哈希、速度原始观测、候选验收：`final_review/results.json`。代码和不含图像/标签内容的结果摘要上传GitHub独立实验分支，原始数据、权重、环境、缓存、凭据不上传。','']
    path=ROOT/'docs/缺陷零改标算法对照_20261112_最终.md';path.write_text('\n'.join(lines),encoding='utf8');print(str(path))
if __name__=='__main__':main()
