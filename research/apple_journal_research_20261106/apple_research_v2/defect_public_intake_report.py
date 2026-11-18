"""Summarize a verified external donor snapshot and draw representative duplicate-screen cases."""

from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from PIL import Image,ImageDraw
from class0_patch_refinement import read,atomic_json,sha
from defect_public_intake import OUT


def report(dst):
    dst=Path(dst).resolve()
    pos=read(dst/"positives.json");receipt=read(dst/"receipt.json")
    assert sha(dst/"positives.json")==receipt["manifest_sha256"]
    dec=read(dst/"image_decisions.json");audit=read(dst/"overlap_audit.json")
    edges=audit["edges"];loss=read(dst/"repeated_lesion_contexts.json")
    byid={r["key"]:r for r in dec}
    labels=sorted({v["category_name"] for v in pos})
    stats={}
    for name in labels:
        matched=[r for r in pos if r["category_name"]==name]
        assert all(r["mask"] and r["raw_split"]=="train" and sha(dst/r["mask"])==r["mask_sha256"] for r in matched)
        stats[name]=dict(regions=len(matched),source_images=len({r["source_id"] for r in matched}),
                         capture_groups=len({r["group_id"] for r in matched}))
    choices=[]
    conditions=[
        ("same capture pair",lambda e:e["rule"]=="same_source_lineage"),
        ("reviewed near-same visible view",lambda e:e["rule"]=="reviewed_near_same_view"),
        ("visually uncertain; one donor view",lambda e:e["rule"]=="reviewed_similar_view"),
        ("same rig; different visible surface",lambda e:(e["foreground"] or {}).get("relationship")=="different_visible_surface"),
        ("shared background only; no confirmed photo match",lambda e: not e["linked"] and (not e["foreground"] or e["foreground"]["relationship"]=="uncertain_surface")),
        ("near-matching visible surface",lambda e:e["rule"]=="verified_near_surface"),
    ]
    for title,test in conditions:
        seq=sorted((e for e in edges if test(e)),key=lambda e:e["phash_distance"]+e["dhash_distance"])
        for e in seq[:2]:choices.append((title,byid[e["a_key"]],byid[e["b_key"]]))
    with ExitStack() as stack:
        zips={v["archive_sha256"]:stack.enter_context(ZipFile(v["archive"])) for v in receipt["source_specs"]}
        if choices:
            out=Image.new("RGB",(660,len(choices)*178),"white");draw=ImageDraw.Draw(out)
            for i,(name,a,b) in enumerate(choices):
                for k,row in enumerate((a,b)):
                    with Image.open(BytesIO(zips[row["archive_sha256"]].read(row["member"]))) as im:
                        im.thumbnail((318,146));out.paste(im,(330*k,178*i))
                    draw.text((330*k,178*i+148),row["member"].split(".rf.")[0][-41:],fill="black")
                draw.text((4,178*i+160),name,fill="black")
            out.save(dst/"foreground_review.jpg",quality=92)
    reasons=Counter(r["decision"] for r in dec)
    edge_types=Counter(
       (e["foreground"] or {}).get("relationship","fixed_scene_background_unconfirmed") if e["rule"]=="perceptual" and not e["linked"]
       else e["rule"] for e in edges)
    checks=dict(class_counts=stats,total_crops=len(pos),source_images=len({r["source_id"] for r in pos}),
                SSDA_ref=audit["SSDA_partitions"],external_seen=len(dec),ssda_matched=sum(e["b_partition"]!="external" for e in edges),
                direct_edge_types=edge_types,decisions=reasons,repeated_lesion_contexts=len(loss),manifest_sha256=receipt["manifest_sha256"])
    atomic_json(dst/"usable_summary.json",checks)
    text=[
        "# 外部果面病灶训练裁片复核",
        "",
        f"两个官方导出包共 {len(dec)} 张照片。当前累计训练清单保留 {len(pos)} 个原作者多边形，来自 {checks['source_images']} 张官方 train 原图。原图、裁片、裁片坐标系的掩膜及原注释 ID 均可追溯。验收集仍为原 SSDA 的 113 图、230 个标注。",
        "",
        "|作者类别|可用病灶|来源原图|采集组|","|---|---:|---:|---:|",
        *[f"|{k}|{v['regions']}|{v['source_images']}|{v['capture_groups']}|" for k,v in stats.items()],
        "",
        "Woo v1 的 378 张原图按作者划分为 266/75/37。train 中有 332 个 scratch。按相对病灶面积覆盖小、中、大实例，逐一核查了 18 个轮廓，均落在果面局部。本批存在单果正反方向与不同光照。hole 和 bruise 未作目标类别。PSU v2 的版本名是 severity_val，实际 ZIP 内 70 图全部列于作者 train。该版本可能原用于严重度验证，在本项目只作训练域辅助；两者均不作为 SSDA 指标的外部测试集。",
        "",
        "PSU 15 个 Codling Moth Entry、13 个 Plum Curculio、13 个 Stink Bug 均经原图、局部放大和多边形叠图逐个核对。后两类的类别名称和果面损伤照片能支持虫伤辅助训练，机构科普资料也描述此类昆虫可造成果面损伤。发布者未提供逐类标注规范，照片不能诊断具体虫种。Plum Curculio 两处有凸起物难以判定，Stink Bug 六处过于弥散，这八处从供体排除；原标注原样保存。Sting、Apple、Apple Scab、Flyspeck 等未作虫伤，其他未标区域不作负例。[Plum Curculio](https://extension.psu.edu/tree-fruit-insect-pest-plum-curculio) · [Stink Bug](https://extension.psu.edu/brown-marmorated-stink-bug)",
        "",
        "## 重复检查",
        "",
        f"重新哈希全体 710 张 SSDA 原图，包括 596 训练、113 开发及 1 张隔离图。本批与 SSDA 的字节/感知筛查命中 {checks['ssda_matched']} 条候选。查重包括缩放、旋转/翻转、中心裁切和轻度斜旋，不能排除所有未知裁片或物理同果关系。",
        "",
        "Woo 有固定检测台及 LEFT/RIGHT 文件时间戳。相同完整采集时间戳只留一张作者 train 代表图。验证或测试图不进入训练。同台背景会在整图感知哈希上制造虚假近邻；比较果面位置、纹理、相对明暗及上下文，只把有前景依据的近似底图候选作为组关联。未证实是同一底图的整台背景关联均在审计中保留，不作独立果实数。不同视角或不同标记可来自同一果，图像/组数不等于独立物理果数。局部供体再次检查图案、尺度和周边果皮，避免跨照片重复计入明显相同的病灶外观。",
        "",
        "另将 82 对最接近的果面或局部像素配对逐一放大复核。19 对果面近似同视图，16 对细节不足以支持重复使用；其余 47 对果面色纹、瑕疵位置或朝向有可见差异。前两组保护措施已计入最终供体清单。一个果面上独立定位的多个病灶分别保留。例如 PSU 图69原作者圈出两处不同位置的虫蛀孔。不能仅通过视觉判断同一物理果实在全体照片中只出现一次。",
        "",
        "|筛查关联或排除|数量|","|---|---:|",*[f"|{k}|{v}|" for k,v in edge_types.items()],
        f"|跨原图高度相似的局部病灶，剔除后项|{len(loss)}|",
        "",
        "[配对照片人工复核](foreground_review.jpg) · [全图保留与排除](image_decisions.json) · [直接配对证据](overlap_audit.json) · [重复局部病灶](repeated_lesion_contexts.json) · [最终正裁片清单](positives.json) · [来源、版本与包哈希](receipt.json)",
    ]
    (dst/"外部病灶入库报告.md").write_text("\n".join(text)+"\n",encoding="utf8")
    print(checks,flush=True)


if __name__=="__main__":
    report(OUT/read(OUT/"active.json")["snapshot"])
