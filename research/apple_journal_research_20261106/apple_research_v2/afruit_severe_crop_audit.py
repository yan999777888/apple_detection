"""Show every severe test image automatically identified as good beside its actual grade crop."""
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "afruit_visible_quality_v1"


def font(px):
    chinese = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(chinese), px) if chinese.is_file() else ImageFont.load_default()


def fit_image(box, width, height):
    return ImageOps.contain(box, (width, height), Image.Resampling.LANCZOS)


def render(rows, title, outfile):
    cols=2
    card_w,card_h=610,263
    board=Image.new("RGB",(cols*card_w,85+((len(rows)+cols-1)//cols)*card_h),"#f7f7f4")
    draw=ImageDraw.Draw(board)
    draw.text((15,10),title,font=font(25),fill="#243a30")
    summary=[]
    for index,row in enumerate(rows):
        left=(index%cols)*card_w
        top=55+(index//cols)*card_h
        path=ROOT/row["path"]
        with Image.open(path) as file:
            im=ImageOps.exif_transpose(file).convert("RGB")
        fruit=row["fruits"][0]
        region=fruit["crop"]
        thumbnail=fit_image(im,284,176)
        w,h=im.size
        td=ImageDraw.Draw(thumbnail)
        td.rectangle([region[0]*thumbnail.width/w,region[1]*thumbnail.height/h,
                      region[2]*thumbnail.width/w,region[3]*thumbnail.height/h],outline="#00ffff",width=2)
        cropped=fit_image(im.crop(region),284,176)
        board.paste(thumbnail,(left+12,top+25))
        board.paste(cropped,(left+304,top+25))
        draw.text((left+12,top),f"{index+1}. {path.stem}",font=font(16),fill="#20372c")
        draw.text((left+12,top+202),"全图判重瑕，框为实际裁取范围" if row["direct_grade"]==2 else
                  f"全图分类 ID {row['direct_grade']}，框为实际裁取范围",font=font(15),fill="#2c4839")
        crop_label = ("好果", "轻瑕果", "重瑕果")[row["linked_grade"]]
        draw.text((left+304,top+202),f"裁片判为{crop_label}",font=font(15),fill="#8f2619")
        frac=(region[2]-region[0])*(region[3]-region[1])/(w*h)
        draw.text((left+304,top+226),f"裁片占原图 {frac:.1%}",font=font(14),fill="#3e4740")
        summary.append(dict(path=row["path"],group_id=row["group"],source_hw=[h,w],crop_xyxy=region,
                            input_area_fraction=frac, source_prediction=row["direct_grade"],
                            cropped_prediction=row["linked_grade"],stage=fruit["maturity"]))
    draw.text((15,board.height-26),"图像：AFruitDB V1，Mojumdar、Mamun、Islam、Hasan，CC BY 4.0，DOI 10.17632/bz65dz2pbj.1。蓝框、裁片与分级文字为本次添加。",font=font(14),fill="#3e4740")
    board.save(OUT/outfile,quality=90)
    return summary


def main():
    rows = json.loads((OUT / "test_predictions.json").read_text(encoding="utf-8-sig"))
    severe = [r for r in rows if r["truth"] == 2]
    errors = [r for r in severe if r["linked_grade"] == 0]
    correct = [r for r in severe if r["linked_grade"] == 2]
    if correct:
        render(correct, "重瑕果识别正确的样例及实际裁片", "重瑕识别正确_全图与裁片.jpg")
    if not errors:
        return
    summary = render(errors, "重瑕果被判为好果的六处结果及实际裁片", "重瑕误放好果_全图与裁片.jpg")
    report = dict(
        cases=summary,group_counts=dict(Counter(r["group_id"] for r in summary)),
        min_area=min(r["input_area_fraction"] for r in summary),max_area=max(r["input_area_fraction"] for r in summary),
        note="Held-out failure analysis only. Cyan rectangle is the actual contextual crop, not a human fruit or defect boundary."
    )
    (OUT / "severe_crop_comparison.json").write_text(json.dumps(report, indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
