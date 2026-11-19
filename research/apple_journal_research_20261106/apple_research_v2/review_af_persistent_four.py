"""Replay only the four author-grade-1 persistently wrong AF development crops."""
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import afruit_grade_crop_alignment as align
import grade_pairs_20261119 as net
from afruit_visible_quality import prepare_input


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/grade_pairs_20261119"
OUT = EXP / "af_persistent_mild_four"
ARMS = ("sparse_verifier_v1", "grounded_sparse_v2", "fruit_surface_burden_v1")
# Manually chosen, illustrative observation points on the visible face. They
# are review notes only, never a trained label or image-based grade override.
PROBES = {
    "165355": [(.64,.55,"梗窝旁深色小斑"),(.50,.75,"下方点状纹理")],
    "165402": [(.55,.69,"下部红黄皮纹"),(.29,.45,"梗窝及果梗")],
    "172449": [(.752,.674,"右侧几道细褐痕"),(.25,.385,"左上弥散色斑"),(.565,.41,"中间竖向色痕")],
    "213541": [(.77,.284,"右上萼端"),(.313,.66,"左下斜褐痕"),(.695,.733,"右下弥散色斑")],
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def patch_box(frame, region):
    h, w = frame.shape[:2]
    cx, cy, size, _ = region
    size = min(max(5, round(size)), max(h, w))
    x = max(0, min(max(0, w-size), round(cx-size/2)))
    y = max(0, min(max(0, h-size), round(cy-size/2)))
    return [x, y, min(w, x+size), min(h, y+size)]


def rgb(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def title(draw, xy, msg, size=20):
    draw.text(xy, msg, font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size), fill="black", stroke_width=0)


def panel(canvas, source, x, y, width, height):
    im = source.copy()
    im.thumbnail((width,height), Image.Resampling.LANCZOS)
    canvas.paste(im,(x+(width-im.width)//2,y+(height-im.height)//2))


def main():
    OUT.mkdir(exist_ok=True)
    table = read(EXP/"development_agreement.json")
    four = [r for r in table["rows"] if r["truth"] == 1 and r["status"] == "wrong_in_all"]
    cached = {r["sha256"]: r for r in read(ROOT/"datasets/grade_pairs_sources_20261119/af_proposals_val.json")["samples"]}
    arms = {arm: {r["sha256"]: r for r in read(EXP/arm/"development_predictions.json")["rows"] if r["panel"] == "auto"} for arm in ARMS}
    assert len(four) == 4
    rows = []
    zoom = Image.new("RGB",(1350,4*320),"white")
    zd = ImageDraw.Draw(zoom)
    for jrow, r in enumerate(four):
        key = r["sha256"]
        local = cached[key]
        assert local["split"] == "val" and local["grade"] == 1
        original = align.read_frame(local)
        raw = prepare_input(original,1280)
        x, y, right, bottom = local["crop"]
        crop = raw[y:bottom,x:right]
        hints = net.map_hints(local["defects"], (x,y), owner=local["fruits"][0]["fruit_id"])
        patches, valid, coords = net.patch_bank(crop,hints)
        assert all(valid) and len(coords) == 8
        boxes = [patch_box(crop,c) for c in coords]
        inputs = {a: arms[a][key] for a in ARMS}
        top = {a: int(np.argmax(t["sparse_scores"])) for a,t in inputs.items()}
        for a, i in top.items():
            assert np.allclose(coords[i][:3],inputs[a]["best_proposal"][:3])
        assert len(set(top.values())) == 1
        best = top[ARMS[0]]
        name = Path(r["path"]).stem
        yrow = jrow*320
        title(zd,(5,yrow+1),name+"　原图局部核查（红圈为观察点）",20)
        probes = []
        source = rgb(original)
        sw,sh = source.size
        fh,fw = raw.shape[:2]
        ch,cw = crop.shape[:2]
        for n,(u,v,label) in enumerate(PROBES[name[-6:]]):
            cx,cy = round(u*cw),round(v*ch)
            sx,sy = round((cx+x)*sw/fw),round((cy+y)*sh/fh)
            extent=round(.14*min(cw*sw/fw,ch*sh/fh))
            sbox=(sx-extent,sy-extent,sx+extent,sy+extent)
            evidence=source.crop(sbox).resize((245,245),Image.Resampling.LANCZOS)
            ed=ImageDraw.Draw(evidence)
            ed.ellipse((113,113,132,132),outline="#df1515",width=1)
            zoom.paste(evidence,(10+440*n,yrow+39))
            hits=[i+1 for i,(l,t,rr,bb) in enumerate(boxes) if l<=cx<rr and t<=cy<bb]
            q=','.join(map(str,hits)) or "无"
            title(zd,(6+440*n,yrow+287),f"{label}　中心落在候选 {q}",16)
            probes.append(dict(label=label,auto_xy=[cx,cy],raw_xy=[sx,sy],raw_window=sbox,candidate_hits=hits))
        sheet = Image.new("RGB",(1450,760),"white")
        d = ImageDraw.Draw(sheet)
        title(d,(16,6),f"{name}    作者 2nd grade（轻瑕）    五版均判"+("重瑕" if r["woo"] == 2 else "好果")+"    "+local["group_id"], 25)
        aw = rgb(raw)
        ImageDraw.Draw(aw).rectangle((x,y,right-1,bottom-1),outline="#00bfa8",width=4)
        crop_img = rgb(crop)
        mapped = crop_img.copy()
        md = ImageDraw.Draw(mapped)
        for i in sorted(range(len(boxes)),key=lambda k:k==best):
            region = boxes[i]
            color = "#ff4b01" if i == best else "#72f5ff"
            l,t,rr,bb = region
            md.rectangle((l,t,rr-1,bb-1),outline=color,width=4)
            md.rectangle((l,t,l+23,t+25),fill=color)
            md.text((l+2,t),str(i+1),font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",20),fill="black")
        for j,(im,label) in enumerate(((aw,"送入流水线的全帧；绿框为实际裁剪"),(crop_img,"实际分级裁片"),(mapped,"8 个候选；橙色为各版共同最高分"))):
            title(d,(j*480+12,51),label)
            panel(sheet,im,j*480+5,85,475,430)
        title(d,(12,527),"候选实际输入（80 × 80，展示放大）　编号 / 局部原始分：稀疏版 → 含上下文版")
        for i,p in enumerate(patches):
            left = i*180+11
            d.rectangle((left,568,left+143,711),outline="#ff4b01" if i==best else "#999999",width=3)
            sheet.paste(Image.fromarray(p).resize((138,138),Image.Resampling.NEAREST),(left+3,571))
            title(d,(left,716),f"{i+1} / {inputs[ARMS[0]]['sparse_scores'][i]:.1f} → {inputs[ARMS[1]]['sparse_scores'][i]:.1f}",17)
        sheet.save(OUT/(name+".jpg"),quality=94)
        rows.append(dict(**r,group=local["group_id"],grid_hw=local["hw"],raw_hw=list(original.shape[:2]),auto_xyxy=local["crop"],
                         external_defects=local["defects"],local_coords=coords,local_boxes_xyxy=boxes,top_index_one_based=best+1,
                         scores={a:t["sparse_scores"] for a,t in inputs.items()},illustrative_observation_points=probes))
    (OUT/"geometry.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),encoding="utf-8")
    pdf=[]
    for row in rows:
        with Image.open(OUT/(Path(row["path"]).stem+".jpg")) as im:
            pdf.append(im.copy())
    zoom.save(OUT/"原图可疑与正常结构局部.jpg",quality=96)
    pdf.append(zoom)
    pdf[0].save(OUT/"四张轻瑕逐图对照.pdf",save_all=True,append_images=pdf[1:],resolution=110.)


if __name__ == "__main__":
    main()
