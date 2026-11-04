"""Render the highest-confidence TRAIN ignore candidates for manual audit."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = Path("C:/Windows/Fonts/msyhbd.ttc")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = [(name, candidate) for name, candidates in payload["records"].items()
            for candidate in candidates]
    rows.sort(key=lambda row: -row[1]["score"])
    rows = rows[:args.limit]
    panel = 360
    sheet = Image.new("RGB", (panel * 4, panel * 2 + 24), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(FONT), 18)
    for step, (name, candidate) in enumerate(rows):
        path = ROOT / "assets/papple/images/train" / name
        image = Image.open(path).convert("RGB")
        x1, y1, x2, y2 = candidate["box"]
        pad = max(x2 - x1, y2 - y1) * .55
        crop_left = max(0, int(x1 - pad))
        crop_top = max(0, int(y1 - pad))
        crop_right = min(image.width, int(x2 + pad))
        crop_bottom = min(image.height, int(y2 + pad))
        crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        scale = min((panel) / crop.width, (panel - 24) / crop.height)
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
                           Image.LANCZOS)
        row, column = divmod(step, 4)
        left = column * panel + (panel - crop.width) // 2
        top = row * panel + 24 + (panel - 24 - crop.height) // 2
        sheet.paste(crop, (left, top))
        local = ((x1 - crop_left) * scale, (y1 - crop_top) * scale,
                 (x2 - crop_left) * scale, (y2 - crop_top) * scale)
        draw.rectangle((left + local[0], top + local[1], left + local[2], top + local[3]),
                       outline=(0, 220, 90), width=3)
        draw.text((column * panel + 6, row * panel + 2),
                  f"{candidate['score']:.3f}  fill={candidate['fill']:.2f}  {name}",
                  font=font, fill=(20, 25, 30))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=94)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
