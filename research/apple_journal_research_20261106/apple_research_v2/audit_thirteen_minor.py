"""Review frozen AFruitDB false good grades, with same-burst controls and fixed-crop ablation."""
import collections
import html
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from afruit_grade_model import ROOT
from afruit_grade_train import read, atomic
from afruit_visible_quality import prepare_input
from matched_grade_loader import load, tensor_at_size


OUT=ROOT/"experiments/grade_failure_audit_20261118/afruit13"
FIELD=ROOT/"experiments/grade_matched_review_v1/field/paired_predictions.json"
PARENT=ROOT/"experiments/afruit_visible_quality_v1/test_predictions_1280.json"
GRADE=ROOT/"experiments/afruit_grade_matched_v1/paired"

# Source-label-independent visual review of the full photograph, actual crop and 224 input.
NOTES={
    '212521':('清楚局部痕迹','中央是深色果脐，右下方另有一个边界清楚的椭圆褐痕，两者在裁片与224输入都可分辨。'),
    '212530':('清楚局部痕迹','左半果皮有一条弯长褐线，右下方另有深色果脐。划线完整留在画面内。'),
    '212548':('此面证据不足','果窝有反光、浅纹及中央暗点，未能从天然果窝中可靠分辨一处损伤；没有标注具体伤处。'),
    '212553':('弱证据或外观边界','右侧近果缘有一处浅褐小记号，与散在果点接近。整图判轻，裁片仍看得见这处记号。'),
    '212954':('弱证据或外观边界','左侧果柄窝附近见褐纹；果皮右上有很淡的细线，未能确定作者判轻依据。'),
    '213105':('此面证据不足','正面接近均匀黄，深色集中于果脐与其阴影，没有清晰独立损伤；轻微浅色差可见。'),
    '213134':('清楚局部痕迹','果柄在左侧，面的中央偏右上有边界清楚的细长浅褐痕；224输入保留该痕。'),
    '213153':('清楚局部痕迹','果脐外，在果面正下缘有一处浅色椭圆凹痕，裁片包含整处。'),
    '213206':('弱证据或外观边界','右半果面有一大片边界较圆的暗色区，轮廓也有局部转折；光照、色泽和压伤从一张图无法区分。'),
    '213223':('清楚局部痕迹','右下缘有小块浅色椭圆痕，果面其余大部分均匀，可见痕貌与213153接近。'),
    '213230':('清楚局部痕迹','左上果皮一条偏红褐的弧线比周围果点长，裁框完整保留，整图和裁片均高分判好。'),
    '213239':('弱证据或外观边界','大部分为连续黄红过渡，左下缘另有一处短暗痕；所在边缘与阴影接近。'),
    '213259':('此面证据不足','果脐在右侧，正面有浅色差和零星微痕，未能指认一个有把握的瑕疵。')}


def image_of(rel):
    return prepare_input(cv2.imdecode(np.fromfile(ROOT/rel,np.uint8),cv2.IMREAD_COLOR),1280)


def tile(im,w,h):
    pil=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    return ImageOps.contain(pil,(w,h),Image.Resampling.LANCZOS)


@torch.inference_mode()
def audit():
    rows=read(FIELD)
    original={r['path']:r for r in read(PARENT)}
    cases=[r for r in rows if r['truth']==1 and r['linked_grade']==0]
    assert len(cases)==13 and {r['group'] for r in cases}=={'burst_0033'}
    controls=[r for r in rows if r['group']=='burst_0033' and r['truth'] in (0,1)
              and r['truth']==r['linked_grade']]
    model,cut,side=load(GRADE,'cpu')
    assert cut==1 and side==224
    out=[]
    (OUT/'images').mkdir(parents=True,exist_ok=True)
    headers=['source + crop bounds','deployed crop','network input geometry']
    for i,r in enumerate(cases+controls):
        old=original[r['path']]
        assert r['same_regions'] and len(old['fruits'])==1
        stem=Path(r['path']).stem
        frame=image_of(r['path'])
        x,y,right,bottom=old['fruits'][0]['crop']
        crop=frame[y:bottom,x:right]
        # This ablation preserves scale, source coordinates and every crop pixel.
        inside=np.full_like(frame,114)
        inside[y:bottom,x:right]=crop
        variants=[frame,crop,inside]
        probs=model(tensor_at_size(variants,'cpu',224)).softmax(1).tolist()
        pred=np.argmax(probs,axis=1).tolist()
        linked=[b for b in old['defects'] if b['association']=='unique']
        out.append(dict(**r,case=i<len(cases),stem=stem,crop_bounds=[x,y,right,bottom],
                    crop_fraction=crop.size/frame.size,linked_lesions=linked,all_lesions=old['defects'],
                    cpu_full_crop_neutral=pred,probabilities=probs))
        src=frame.copy();cv2.rectangle(src,(x,y),(right-1,bottom-1),(0,210,255),4)
        board=Image.new('RGB',(816,314),'white')
        draw=ImageDraw.Draw(board)
        for j,im in enumerate((src,crop,cv2.resize(crop,(max(1,round(224*crop.shape[1]/max(crop.shape[:2]))),max(1,round(224*crop.shape[0]/max(crop.shape[:2])))),interpolation=cv2.INTER_AREA))):
            if j==2:
                v=np.full((224,224,3),(104,116,124),np.uint8);hh,ww=im.shape[:2];xx=(224-ww)//2;yy=(224-hh)//2;v[yy:yy+hh,xx:xx+ww]=im;im=v
            pic=tile(im,264,241)
            board.paste(pic,(j*272+(272-pic.width)//2,16+(241-pic.height)//2))
            draw.text((j*272+5,258),headers[j],fill='black')
        draw.text((5,280),f'{i+1:02d} {stem} GT {r["truth"]} | original/model crop {r["previous_linked"]}/{r["linked_grade"]}, model whole {r["direct_grade"]}',fill='black')
        board.save(OUT/'images'/f'{stem}.jpg',quality=90)
    atomic(OUT/'rows.json',out)
    for label,part in [('miss',out[:13]),('mild', [r for r in out if not r['case'] and r['truth']==1]),('good',[r for r in out if not r['case'] and r['truth']==0])]:
        print(label,len(part),'alerts',sum(bool(r['linked_lesions']) for r in part),'ablation',collections.Counter(tuple(r['cpu_full_crop_neutral']) for r in part))
        display=part if label=='miss' else [part[round(n*(len(part)-1)/7)] for n in range(8)]
        for offset in range(0,len(display),4):
            sub=display[offset:offset+4]; page=Image.new('RGB',(816,314*len(sub)),'white')
            for j,r in enumerate(sub):
                with Image.open(OUT/'images'/f'{r["stem"]}.jpg') as im: page.paste(im,(0,j*314))
            page.save(OUT/f'{label}_{offset+1:02d}.jpg',quality=88)


def local_matching(rows):
    # Interior-only correspondences avoid the recurring white tile background.
    detector=cv2.SIFT_create(nfeatures=650,contrastThreshold=.012)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    descriptors=[]
    for r in rows:
        x,y,right,bottom=r['crop_bounds']
        im=cv2.resize(image_of(r['path'])[y:bottom,x:right],(440,440),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        mask=np.zeros_like(gray);cv2.ellipse(mask,(220,220),(146,146),0,0,360,255,-1)
        keypoints,desc=detector.detectAndCompute(gray,mask)
        descriptors.append((np.float32([k.pt for k in keypoints]),desc))
    pairs=[]
    for i,r in enumerate(rows):
        if not r['case']:
            continue
        q,dq=descriptors[i]
        for j,other in enumerate(rows):
            if j==i or (other['case'] and j<i):
                continue
            p,dp=descriptors[j]
            matches=[a for a,b in matcher.knnMatch(dq,dp,k=2) if a.distance<.72*b.distance]
            if len(matches)<12: continue
            a=np.float32([q[m.queryIdx] for m in matches]);b=np.float32([p[m.trainIdx] for m in matches])
            cv2.setRNGSeed(1209)
            mat,flags=cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3)
            if mat is None: continue
            selected=flags.ravel().astype(bool);nin=int(selected.sum())
            if nin<12: continue
            def spread(v):
                return float(cv2.contourArea(cv2.convexHull(v[selected]))/440**2)
            support=min(spread(a),spread(b))
            if support>=.035:
                pairs.append(dict(image=r['stem'],other=other['stem'],other_truth=other['truth'],count=nin,
                                  putative=len(matches),relative_hull=support))
    atomic(OUT/'interior_matches.json',pairs)
    print("Interior geometric matches",[(r['image'][-6:],r['other'][-6:],r['count']) for r in pairs])


def publish():
    rows=read(OUT/'rows.json')
    matches=read(OUT/'interior_matches.json')
    cases=[r for r in rows if r['case']]
    assert len(cases)==len(NOTES)==13
    grade_name=['好','轻','重']
    lines=[]
    content=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AFruitDB 十三张轻瑕误放与同批对照</title><style>body{max-width:1000px;margin:1.5rem auto;padding:0 1rem;font:16px/1.5 system-ui}article{border-top:1px solid #888;padding:.75rem 0}img{max-width:100%;height:auto}summary{cursor:pointer}</style><h1>十三张轻瑕误放与同批对照</h1><p>从左到右是原图及黄线裁框、实际裁片、分级器224输入。图片底部 GT/预测使用 0 好、1 轻、2 重。所有原图来自 AFruitDB（Mojumdar 等，CC BY 4.0），裁片与预测由本项目添加。目视属性不是专家诊断。</p>']
    record=[]
    for i,r in enumerate(cases,1):
        suffix=r['stem'][-6:]
        category,note=NOTES[suffix]
        full,crop,neutral=[grade_name[x] for x in r['cpu_full_crop_neutral']]
        row=dict(path=r['path'],timestamp=suffix,visual_category=category,visual_notes=note,
                 full_grade=full,crop_grade=crop,neutral_grade=neutral,
                 linked_alerts=len(r['linked_lesions']),crop_bounds=r['crop_bounds'])
        record.append(row)
        lines.append(f"| {suffix} | {category} | {note} | {full}/{crop}/{neutral} | {row['linked_alerts']} |")
        content.append(f'<article><h2>{i:02d} · {r["stem"]} · {category}</h2><img loading="lazy" alt="同一张误判原图及输入" src="images/{r["stem"]}.jpg"><p>{html.escape(note)}</p><p>整图 / 裁片 / 框外改灰：{full} / {crop} / {neutral}。已归属瑕疵告警：{row["linked_alerts"]}。</p></article>')
    for c in (1,0):
        controls=[r for r in rows if not r['case'] and r['truth']==c]
        content.append(f'<details><summary>同批判对的{grade_name[c]}果 · {len(controls)} 张</summary>')
        content.extend(f'<article><b>{r["stem"]}</b><br><img loading="lazy" alt="同批对照照片" src="images/{r["stem"]}.jpg"></article>' for r in controls)
        content.append('</details>')
    (OUT/'逐张核查.html').write_text(''.join(content)+'</html>',encoding='utf8')
    atomic(OUT/'visual_review.json',record)
    atomic(OUT/'summary.json',dict(n_total=len(rows),n_misses=len(cases),primary_visual_counts=dict(collections.Counter(v[0] for v in NOTES.values())),
           model_cpu_vs_gpu_disagreements=sum(r['cpu_full_crop_neutral'][:2]!=[r['direct_grade'],r['linked_grade']] for r in rows),
           neutral_intervention=dict(cpu_order="whole/crop/neutral", patterns=dict(collections.Counter('/'.join(map(str,r['cpu_full_crop_neutral'])) for r in cases))),
           lesion_alert_images={name:sum(bool(r['linked_lesions']) for r in rows if predicate(r)) for name,predicate in (
               ('missed_minor',lambda r:r['case']),('correct_minor',lambda r:not r['case'] and r['truth']==1),
               ('correct_good',lambda r:not r['case'] and r['truth']==0))},supported_near_pairs=matches))
    lead="""# AFruitDB 的 13 张轻瑕果误判好果

从 110 张历史回归照片中取出作者标为轻瑕、自动链路恰好检出一果且最终判好的全部 13 张。13 张均属于 2023 年 9 月 30 日的同一白色台面连拍组，原版也都判好。对照为同组判对的 14 张轻瑕及 39 张好果。所有 66 张图片均对照检测器实际边框、10% 扩框裁片及 224 输入重建；整图及裁片的 CPU FP32 判级与之前记录的 GPU 判级在这 66 张上全部一致。这是失败归因材料，用于训练的样本来自另外的训练批次。

13 张中，每个框均围到这颗苹果，并保留了目视能指认的异常，没有发现定位到其他物体或把伤处切掉的案例。6 张仍能辨认独立局部痕迹，4 张只有较弱提示或颜色边界，3 张在此面无法可靠指认瑕疵。未发现主导性的整体运动模糊。该划分由本地模型逐图目视，不能替代果品专家复核；作者只标整张照片的等级，没有标出具体伤处或物理果实 ID。证据弱的 3 张不能据此改标签。小范围采集中，个别相邻视图形态相似；使用果面内部特征点和仿射几何一致检验，没有一对超过预设门槛，既不能认定是重复照片，也不能证明是 13 个独立实物。

对相同权重做了输入对照。每项按“整图 / 自动裁片 / 保留原果尺寸和位置但框外像素统一灰化”的顺序列出。8 张三路都判好，单纯恢复整图也无法救回。其余 5 张整图判轻、裁片判好，其中 4 张仅灰化框外就转为好，说明这些图的整图判定对框外内容敏感；1 张灰化后仍判轻，缩放、位置、填边中的一个或多个因素仍有影响。灰化本身会改变图像分布，该诊断没有证明模型只依赖桌面。

同批对照也有自然果点、果窝和浅表纹。例如好果 212459 表皮有两小段褐线，好果 213017 有单个褐点。不能把所有暗点一律当轻瑕。原可见缺陷模型在 8/13 张漏分轻瑕中给出了至少一个归属该果的告警，在 11/14 张判对轻瑕、9/39 张判对好果上也给了告警。AFruitDB 无局部瑕疵真值，告警本身既不证明伤处，也不适合直接改等级。

| 文件时刻 | 可见证据 | 逐图观察 | 整/裁/灰 | 同果告警数 |
|---|---|---|---|---:|
"""
    dest=ROOT/'docs/AFruitDB十三张轻瑕错例逐果核查_20261118.md'
    dest.write_text(lead+'\n'.join(lines)+'\n\n[全部错例和同批对照图](../experiments/grade_failure_audit_20261118/afruit13/逐张核查.html) · [实际判级和裁框](../experiments/grade_failure_audit_20261118/afruit13/rows.json) · [对照实现](../apple_research_v2/audit_thirteen_minor.py)\n',encoding='utf8')


if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--match',action='store_true',help='Check repeated views across the reviewed cases')
    cli.add_argument('--publish',action='store_true',help='Attach case-by-case visual findings and write the review')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    args=cli.parse_args()
    if args.match:
        local_matching(read(OUT/'rows.json'))
    elif args.publish:
        publish()
    else:
        audit()
