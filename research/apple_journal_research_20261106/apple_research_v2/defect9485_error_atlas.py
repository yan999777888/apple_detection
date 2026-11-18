"""Display all class-1 false alarms far from existing annotations, with fruit context."""

import math
from PIL import Image, ImageDraw
from class0_patch_refinement import ROOT, manifest, read, atomic_json, configure
from audit_defect_causes_20261112 import match

OUT = ROOT / "experiments/defect9485_error_atlas"


def compare_annotated(im, prediction, gt, code):
    candidates = [g["box"] for g in gt if g["class"] == 1]
    if not candidates:
        return None

    def diameter(box):
        return math.sqrt((box[2] - box[0]) * (box[3] - box[1]))

    ref = min(
        candidates, key=lambda box: abs(math.log(diameter(prediction) / diameter(box)))
    )
    panel = Image.new("RGB", (184, 115), "white")
    ImageDraw.Draw(panel).text((2, 2), code + "        GT1", fill="black")
    for j, box in enumerate((prediction, ref)):
        x = int((box[0] + box[2]) / 2)
        y = int((box[1] + box[3]) / 2)
        panel.paste(im.crop((x - 44, y - 44, x + 44, y + 44)), (j * 92, 21))
    return panel


def main():
    configure(OUT, "cpu")
    rows = read(ROOT / "experiments/class0_crop_deploy_v1/fp32/dev_predictions.json")
    lookup = manifest()
    labels = []
    panels = []
    pairs = []
    for r in rows:
        gt = r["GT"]
        pred = r["pred"]
        assigned, ov = match(pred, gt)
        for i, p in enumerate(pred):
            if p[5] != 1 or i in assigned or (len(gt) and ov[i].max() >= 0.1):
                continue
            code = f"B{len(labels) + 1:02}"
            with Image.open(lookup[r["image_id"]]["rgb_path"]) as src:
                im = src.convert("RGB")
            pair = compare_annotated(im, p, gt, code)
            if pair is not None:
                pairs.append(pair)
            x, y, X, Y = p[:4]
            cx = (x + X) / 2
            cy = (y + Y) / 2
            side = max(64, 2 * max(X - x, Y - y))
            l = int(cx - side / 2)
            t = int(cy - side / 2)
            side = int(side)
            crop = im.crop((l, t, l + side, t + side)).resize((154, 154))
            thumb = im.resize((154, 154))
            dr = ImageDraw.Draw(thumb)
            w, h = im.size
            for g in gt:
                b = g["box"]
                dr.rectangle(
                    (154 * b[0] / w, 154 * b[1] / h, 154 * b[2] / w, 154 * b[3] / h),
                    outline="lime",
                )
            dr.rectangle(
                (154 * x / w, 154 * y / h, 154 * X / w, 154 * Y / h),
                outline="magenta",
                width=2,
            )
            panel = Image.new("RGB", (310, 178), "white")
            panel.paste(thumb, (0, 23))
            panel.paste(crop, (156, 23))
            ImageDraw.Draw(panel).text(
                (3, 4), f"{code}    class 1    score {p[4]:.2f}", fill="black"
            )
            panels.append(panel)
            labels.append(
                dict(
                    code=code,
                    image=r["image_id"],
                    prediction=p,
                    nearest_iou=float(ov[i].max()) if len(gt) else 0,
                )
            )
    for j in range(0, len(panels), 8):
        sheet = Image.new("RGB", (1240, 356), "white")
        for k, p in enumerate(panels[j : j + 8]):
            sheet.paste(p, ((k % 4) * 310, (k // 4) * 178))
        sheet.save(OUT / f"sheet{j // 8 + 1}.jpg", quality=94)
    if pairs:
        sheet = Image.new("RGB", (920, 115 * ((len(pairs) + 4) // 5)), "white")
        for j, panel in enumerate(pairs):
            sheet.paste(panel, (184 * (j % 5), 115 * (j // 5)))
        sheet.save(OUT / "same_scale_annotated_vs_unmarked.jpg", quality=94)
    atomic_json(OUT / "index.json", labels)
    print(len(labels))


if __name__ == "__main__":
    main()
