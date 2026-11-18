"""Review the complete original validation minor class and random good controls."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
import afruit_grade_model as grade
import afruit_grade_crop_alignment as aligned


ROOT = grade.ROOT
OUT = ROOT / "experiments/afruit_grade_matched_v1"


def board():
    rows=aligned.split_rows("val")
    crops=aligned.load_cache("val",rows)
    ref=json.loads((aligned.OUT/"paired_val_predictions.json").read_bytes())["crop_aligned"]
    by_sha={row["sha256"]:row for row in ref}
    good=[i for i,r in enumerate(rows) if r["grade"]==0]
    controls=np.random.default_rng(1826).choice(good,size=4,replace=False)
    chosen=[i for i,r in enumerate(rows) if r["grade"]==1]+sorted(controls.tolist())
    target=Image.new("RGB",(5*330,4*194),"#f3f1ec")
    draw=ImageDraw.Draw(target)
    names=("good","minor","severe","unresolved")
    for pos,i in enumerate(chosen):
        r=rows[i]; row=by_sha[r["sha256"]]
        x,y=330*(pos%5),194*(pos//5)
        rgb=cv2.cvtColor(aligned.read_frame(r),cv2.COLOR_BGR2RGB)
        full=Image.fromarray(rgb)
        full.thumbnail((156,148))
        target.paste(full,(x+3+(156-full.width)//2,y+4+(148-full.height)//2))
        if crops[i] is not None:
            crop=Image.fromarray(crops[i]).resize((148,148),Image.Resampling.LANCZOS)
            target.paste(crop,(x+165,y+4))
        draw.text((x+3,y+154),f'{Path(r["path"]).stem}  {r["group_id"]}',fill="#172129")
        draw.text((x+3,y+172),f'GT {names[r["grade"]]}   whole {names[row["whole"]]}   crop {names[row["auto"]]}',fill="#172129")
    target.save(OUT/"validation_minor_and_controls.jpg",quality=91)


if __name__=="__main__":
    board()
