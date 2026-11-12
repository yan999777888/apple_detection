"""Deterministic evidence report; custom deployment AP explicitly distinct from native AP."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/defect_algorithm_steps_20261112_v2'
sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import atomic_json

def deployment_ap(records):
    import numpy as np
    from audit_defect_causes_20261112 import ious
    values={}
    for c in (0,1):
        gt={r['image_id']:[g['box'] for g in r['GT'] if g['class']==c] for r in records};total=sum(map(len,gt.values()))
        pp=sorted([(float(p[4]),ri,pi,r['image_id'],p[:4]) for ri,r in enumerate(records) for pi,p in enumerate(r['pred']) if int(p[5])==c],key=lambda p:(-p[0],p[1],p[2]))
        vals=[]
        for t in np.arange(.5,.951,.05):
            used={k:set() for k in gt};tp=[]
            for score,ri,pi,image,box in pp:
                available=[j for j in range(len(gt[image])) if j not in used[image]];ok=False
                if available:
                    overlaps=ious([box],[gt[image][j] for j in available])[0];best=int(overlaps.argmax())
                    if overlaps[best]>=t:used[image].add(available[best]);ok=True
                tp.append(int(ok))
            cum=np.cumsum(tp);rec=cum/max(total,1);prec=cum/np.arange(1,len(tp)+1)
            ap=np.mean([max(prec[rec>=r],default=0) for r in np.linspace(0,1,101)]);vals.append(float(ap))
        values[str(c)]={'AP50':vals[0],'AP50_95':float(np.mean(vals)),'AP_per_IoU':vals,'GT':total}
    return {'per_class':values,'mAP50':sum(x['AP50'] for x in values.values())/2,'mAP50_95':sum(x['AP50_95'] for x in values.values())/2,'definition':'custom class-aware greedy score-order IoU matching, 101-point precision envelope; NOT official Ultralytics AP; same 113 images/230 GT, candidates>=.001 after explicit NMS or declared fusion'}

def read(path):return json.loads(path.read_text(encoding='utf8'))
def fmt(m):
    t=m['total'];return f"{t['TP']} / {t['FP']} / {t['FN']} | {100*t['P']:.2f}% | {100*t['R']:.2f}% | {100*m['class']['0']['R']:.2f}% | {100*m['class']['1']['R']:.2f}% | {m['large']['TP']}/{m['large']['GT']}"

def main():
    from defect_algorithm_modes_20261112 import combine
    import numpy as np
    phase=read(BASE/'phase1/results.json');source=read(BASE/'phase1/source_predictions.json');parity=BASE/'mode_deployment_validation/results.json';verified=read(parity)['modes'] if parity.exists() else None
    lines=['# 原强模型缺陷检测：零改标算法对照','', '本轮执行用户于 2026-11-12 授权的全部算法步骤。准确度优先，双任务部署目标约 20 FPS；不要求以牺牲召回来追求更高帧率。全部文件、环境和缓存留在 F 盘。','',
      '## 实验边界与预先固定的方案','',
      '- 保留原全局 Base＋Small，以及当前缺陷强权重 `composite_frozen.pt`（SHA256 `f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6`）。本轮不是替换为弱学生模型。',
      '- 原训练 596 图、开发 113 图/230 个缺陷 GT 均不改标；不使用封存测试集；类别维持原数值 ID，不能把未经作者确认的名称当成确定语义。',
      '- 旧报告 B 节的人工复核/重标安排按本次授权跳过。疑似暗斑/划痕不等于专家确认的真实缺陷，Ignore Shield 是待检验的鲁棒训练假设。',
      '- 实际本地 TAL 包含按正样本归一化的 alignment metric，不能直接把 IoU^6 当作最终分类目标，也不能由此证明低分漏检的因果关系。one2many 与 one2one 的监督数量差异也不能笼统保证十倍。',
      '- WBF 只融合同类且 IoU>0.55 的重叠框，不把不相交的相邻划痕强行合并。当前实现为明确记录公式的 WBF-style 对照，不宣称与某个第三方库逐位相同。',
      '- 速度范围：640、FP32、成熟度分割＋缺陷检测，两任务从内存 BGR 图像到 CPU Results；不含磁盘读取、相机、深度或机械臂。不能称实机三维整链路 20Hz。','',
      '## 1. 无训练的完整开发集对照','',
      '工作点 conf=0.25，同类 NMS=0.7；WBF IoU=0.55，输入候选最低分 0.001。双头/翻转融合的置信度为簇内成员均值乘来源支持比例，低分成员可能拉低输出分数，结果不应泛化为所有 WBF 配方均无效。','',
      '| 方案 | TP / FP / FN | P | R | class0 R | class1 R | 大框 TP/GT | 均时 / FPS | P95 |','|---|---|---|---|---|---|---|---|---|']
    ap={}
    for name,r in phase['modes'].items():
        m=verified[name] if verified else r;s=r['speed'];lines.append(f"| {name} | {fmt(m)} | {s['mean_ms']:.2f} ms / {s['serial_fps']:.2f} | {s['p95_ms']:.2f} ms |")
        rows=[{'image_id':r['image_id'],'GT':r['GT'],'pred':combine({k:np.asarray(v,dtype=np.float32).reshape(-1,6) for k,v in r['sources'].items()},name,(.001,.001)).tolist()} for r in source];ap[name]=deployment_ap(rows)
    lines+=['','每个速度为 40 张固定图、预热 5 帧、3 轮，共 120 次，属于短程基准。不能把它和旧 1200 次持续测试混为一谈。',
      '实际部署 batch 形状逐模式复核：'+('已完成；详细逐 GT 差异见 mode_deployment_validation。' if verified else '待完成。缓存采集含 batch2 原图/翻转，不默认认为其与 batch1 逐位一致。'),'',
      '## 2. 训练集 Ignore Shield','']
    d=read(BASE/'shield_prepare/defect_train_ignore.json');checks=read(BASE/'cpu_tests/results.json')
    lines+=[f"原强模型在训练集的候选，conf≥0.15 且与所有 GT 最大 IoU<0.1：{d['images_with_candidates']} 张图、{d['candidate_count']} 个区域。它们来自训练内预测，不冒充 OOF，也不新增伪正标签。",'',
      f"{checks['passed']} 项 CPU 检查通过。仅屏蔽忽略区内、未分配给任何 GT 且不落入任何 GT 内的背景锚点 BCE；正样本梯度、框回归损失不变。GT 与忽略框共享精确的缩放/平移/翻转，忽略框不进入目标列表。",'',
      '## 3. 真正的训练内三折 OOF 与类别阈值','',
      '同一采集秒 `same_second_group` 不跨折。三折分别从官方 COCO yolo26n 初始化，固定训练 40 轮，取最后一轮而非验证最优。禁止从已见过全部训练图的当前强权重生成“OOF”。物理果实 ID 不明，仍不能宣称完全同果独立。',
      '每类阈值仅在 OOF 上从 0.01–0.60（步长0.01）选择：FP 不超过该类 OOF 在0.25下的FP，且P不下降，优先最大TP，再最小FP、最接近0.25。阈值落盘冻结后才读取开发集预测；开发集不参与选择。','']
    oof=BASE/'oof_v2/results.json'
    if oof.exists():
        o=read(oof);lines += [f"完成，冻结阈值：{o['thresholds']}。",'', '| 工作点 | TP / FP / FN | P | R | class0 R | class1 R | 大框 TP/GT |','|---|---|---|---|---|---|---|',f"| 原0.25 | {fmt(o['dev_baseline'])} |",f"| OOF冻结阈值 | {fmt(o['dev_selected'])} |"]
    else:lines+=['进行中，尚无完整 OOF 阈值结论。']
    lines+=['','## 4. 六组受控微调','',
      '全部从同一个原强 checkpoint 开始，成熟度分支冻结；FP32、640、AdamW lr=1e-5、每组3轮/114步、每轮596次抽样、microbatch4/有效batch16，固定最后一轮。原始GT文件不改。',
      '全部六组共用新的显式仿射增强路径以确保忽略框与GT严格对齐：平移±0.1、翻转0.5、亮度±0.1、mosaic关闭。G0尺度0.75–1.25，尺度实验0.5–1.3；所以与旧训练增强不能称完全相同，但本轮各单因素有自己的G0。',
      'G0=继续训练对照；G1=仅Ignore Shield；G2=仅class0或大框图像2.5倍抽样概率；G3=仅尺度0.5–1.3；G4=仅TAL beta6→3；G5=四项组合。过采样使用固定596次有放回抽样，不扩张总更新预算。按原建议的“或”选择TAL路线，未叠加另一种回归损失。','']
    training=BASE/'train/results.json'
    if training.exists():
        t=read(training);lines+=['| 组 | TP / FP / FN | P | R | class0 R | class1 R | 大框 TP/GT | FPS | P95 ms |','|---|---|---|---|---|---|---|---|---|']
        for arm,r in t['arms'].items():
            lines.append(f"| {arm} | {fmt(r['fixed'])} | {r['speed']['serial_fps']:.2f} | {r['speed']['p95_ms']:.2f} |")
            ap[arm]=deployment_ap(read(BASE/'train'/arm/'predictions.json'))
        lines+=['','全部完成。原始强权重与GT哈希复核未变；逐GT新增漏检、恢复目标、原生AP以及权重路径见各组 result.json。']
    else:lines+=['尚未完成，不以小批量检查当作正式训练结果。']
    lines+=['','## 5. 指标与结论边界','',
      '- 固定工作点 P/R 以框 IoU≥0.5 匹配。这里是缺陷框任务，不能与原全局苹果 amodal 框94.43%或局部分割AP混比。',
      '- `deployment_ap.json`另给显式NMS/声明融合后的自定义101点AP；匹配算法与Ultralytics原生实现不同，严格分列。',
      '- 只有在完整开发集与原强基线比较后才能判断收益。召回增加但误检增加，或P升高但漏检增多，均需明确披露。不会因某一个数变高自动覆盖原模型。',
      '- 公开数据室内域、标注定义与真实果园差异仍在；纯算法实验无法证明所有疑似漏标已经解决。','',
      '## 工程审计记录','',
      '- phase1 v1 的原图 one2many 缓存被翻转分支覆盖，已声明作废并用 v2 完整重推；未据此晋级。',
      '- 初次 OOF 使用日期 split_group 导致训练组为空，在数据审计前失败、没有训练。修为明确的同秒组，另存 oof_v2，失败记录保留。',
      '- 不覆盖旧运行、不上传权重/原数据/缓存/环境/凭据；代码与可分享证据在独立 experiment/journal-defect-20261112 分支管理。','']
    atomic_json(BASE/'deployment_ap.json',ap)
    (ROOT/'docs/缺陷零改标算法对照_20261112.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'report':'docs/缺陷零改标算法对照_20261112.md','phase1':True,'OOF_complete':oof.exists(),'training_complete':training.exists()}))
if __name__=='__main__':main()
