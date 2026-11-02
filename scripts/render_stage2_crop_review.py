"""Render saved round-1 crop detections only; no inference, GT or mask synthesis."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CASES = ("crop512_nms080_k64", "crop512_nms075_k16")
LABELS = ("NMS 0.80 / K64", "NMS 0.75 / K16")
COLORS = ((29,210,255),(255,192,35))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def font(size):
    for path in (Path("C:/Windows/Fonts/arial.ttf"),Path("C:/Windows/Fonts/segoeui.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path),size)
    return ImageFont.load_default(size=size)


def iou(a,b):
    intersect=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersect
    return intersect/union if union else 0


def annotate(crop, frame, color, label):
    panel=Image.new("RGB",(512,590),(24,30,38))
    panel.paste(crop,(0,64))
    draw=ImageDraw.Draw(panel)
    draw.text((12,7),frame["image_id"],font=font(21),fill="white")
    draw.text((12,35),f"{label} | boxes={frame['selected_count']} | conf>=0.70",font=font(17),fill=color)
    for index,(box,score) in enumerate(zip(frame["endpoint_boxes"],frame["endpoint_scores"],strict=True)):
        x1,y1,x2,y2=box
        coords=(x1,y1+64,min(x2,511),min(y2,511)+64)
        draw.rectangle(coords,outline=color,width=3)
        text=f"{index+1}: {score:.3f}"
        bounds=draw.textbbox((0,0),text,font=font(18))
        tw,th=bounds[2]-bounds[0]+8,bounds[3]-bounds[1]+8
        tx=max(0,min(x1,512-tw));ty=max(64,min(y1+64-th,576-th))
        draw.rectangle((tx,ty,tx+tw,ty+th),fill=(10,18,25))
        draw.text((tx+4,ty+2-bounds[1]),text,font=font(18),fill=color)
    if not frame["endpoint_boxes"]:
        draw.rectangle((10,78,286,110),fill=(10,18,25))
        draw.text((18,83),"NO SAVED OUTPUT >= 0.70",font=font(18),fill="white")
    return panel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    directory=args.benchmark_dir.resolve();out=(args.output or directory/"review").resolve()
    if directory.drive.upper()!="F:" or out.drive.upper()!="F:":
        raise ValueError("Files must remain on F:")
    out.mkdir(parents=True,exist_ok=False)
    (out/"single").mkdir();(out/"pairs").mkdir()
    source=directory/"benchmark_results.json"
    status=directory/"benchmark_status.json"
    receipts=[]
    def bind(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        if expected is not None and digest!=expected:
            raise ValueError("Input SHA differs: "+str(path))
        receipts.append({"path":str(path),"sha256":digest,"bytes":path.stat().st_size})
        return digest
    bind(Path(__file__));bind(source);bind(status)
    data=json.loads(source.read_text(encoding="utf-8"))
    state=json.loads(status.read_text(encoding="utf-8"))
    if data!=state or data.get("phase")!="complete" or data.get("final_threshold")!=.7:
        raise ValueError("Completed benchmark/result snapshot or final threshold differs")
    samples=data["samples"]
    if len(samples)!=8 or len({s["image_id"] for s in samples})!=8:
        raise ValueError("Exactly eight distinct benchmark samples required")
    frames={}
    for case in CASES:
        bind(directory/(case+".json"))
        stage=json.loads((directory/(case+".json")).read_text(encoding="utf-8"))
        if stage!=data["cases"][case] or stage["phase"]!="complete":
            raise ValueError("Per-case benchmark snapshot differs")
        selected=[f for f in stage["frames"] if f["round"]==1]
        if [f["image_id"] for f in selected]!=[s["image_id"] for s in samples]:
            raise ValueError("First-round sample identity/order differs")
        frames[case]=selected
    panels=[];rows=[];outputs=[]
    for index,sample in enumerate(samples):
        path=Path(sample["RGB_path"])
        bind(path,sample["sha256"])
        with Image.open(path) as original:
            rgb=original.convert("RGB")
        if [rgb.height,rgb.width]!=sample["native_hw"]:
            raise ValueError("Native RGB shape differs")
        expected_crop=[(rgb.width-512)//2,(rgb.height-512)//2,(rgb.width-512)//2+512,(rgb.height-512)//2+512]
        current=[];details=[]
        for case,label,color in zip(CASES,LABELS,COLORS,strict=True):
            frame=frames[case][index]
            boxes,scores=frame["endpoint_boxes"],frame["endpoint_scores"]
            if frame["crop_box_in_original"]!=expected_crop or frame["native_output_hw"]!=[512,512] or frame["input_hw"]!=[512,512]:
                raise ValueError("Saved frame does not represent the native centre crop")
            if len(boxes)!=len(scores) or len(boxes)!=frame["selected_count"]:
                raise ValueError("Endpoint count differs")
            for box,score in zip(boxes,scores,strict=True):
                if len(box)!=4 or not all(math.isfinite(x) for x in box) or not 0<=box[0]<box[2]<=512 or not 0<=box[1]<box[3]<=512 or not .7<=score<=1:
                    raise ValueError("Invalid crop-coordinate box/score")
            crop=rgb.crop(expected_crop)
            panel=annotate(crop,frame,color,label)
            filename=out/"single"/f"{index+1:02d}_{Path(sample['image_id']).stem}_{case}.png"
            panel.save(filename);outputs.append(filename)
            current.append(panel)
            overlaps=[{"a":a+1,"b":b+1,"box_iou":iou(boxes[a],boxes[b])} for a in range(len(boxes)) for b in range(a+1,len(boxes))]
            details.append({"case":case,"box_count":len(boxes),"endpoint_boxes":boxes,"endpoint_scores":scores,
                            "maximum_pair_box_iou":max((x["box_iou"] for x in overlaps),default=None),
                            "pairs_box_iou_ge_0_70":[x for x in overlaps if x["box_iou"]>=.7],
                            "note":"Overlapping boxes alone do not establish duplicate apples or detection correctness"})
        pair=Image.new("RGB",(1040,640),(24,30,38))
        pair.paste(current[0],(0,0));pair.paste(current[1],(528,0))
        ImageDraw.Draw(pair).text((12,594),"Saved round 1 | original Small weights | true 512 crop | boxes only, no mask/GT",font=font(20),fill="white")
        ImageDraw.Draw(pair).text((12,620),"Speed diagnostic only; not D405 tracking or crop accuracy validation",font=font(16),fill=(190,199,208))
        pairpath=out/"pairs"/f"{index+1:02d}_{Path(sample['image_id']).stem}_comparison.png"
        pair.save(pairpath);outputs.append(pairpath)
        panels.append(current)
        identical=all(frames[CASES[0]][index][key]==frames[CASES[1]][index][key] for key in ("endpoint_boxes","endpoint_scores"))
        rows.append({"image_id":sample["image_id"],"RGB_path":str(path),"RGB_sha256":sample["sha256"],
                     "crop_box_in_original":expected_crop,"round":1,"cases":details,"case_endpoints_exact":identical})
    for page in range(2):
        canvas=Image.new("RGB",(1088,2540),(18,23,30));draw=ImageDraw.Draw(canvas)
        draw.text((24,18),f"Native 512 crop: saved detections   |   {page+1}/2",font=font(29),fill="white")
        draw.text((24,60),"Original Small / FP16 / conf >= 0.70 / first timed round",font=font(21),fill=(196,206,219))
        draw.text((24,92),"Left: NMS .80 K64     Right: NMS .75 K16",font=font(22),fill=(196,206,219))
        draw.text((24,126),"Boxes only. No rerun, GT scoring, invented contour, or D405 tracking claim.",font=font(19),fill=(196,206,219))
        for offset in range(4):
            index=page*4+offset
            canvas.paste(panels[index][0],(24,168+offset*590))
            canvas.paste(panels[index][1],(552,168+offset*590))
        file=out/f"comparison_page_{page+1:02d}.png"
        canvas.save(file);outputs.append(file)
    for receipt in receipts:
        if sha(receipt["path"])!=receipt["sha256"]:
            raise RuntimeError("Input changed while rendering")
    manifest={"schema":"stage2_saved_crop_detection_review_v1","completed_utc":datetime.now(timezone.utc).isoformat(),
              "GPU_used":False,"inference_performed":False,"GT_accuracy_evaluated":False,"mask_rendered":False,
              "images":8,"round":1,"threshold":.7,"sources_before_after_identical":True,"input_receipts":receipts,
              "sample_reviews":rows,"outputs":[{"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size} for p in outputs]}
    (out/"review_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),"manifest_sha256":sha(out/"review_manifest.json"),
                      "images":8,"PNGs":len(outputs),"samples":[{"image_id":r["image_id"],"counts":[x["box_count"] for x in r["cases"]],"exact":r["case_endpoints_exact"]} for r in rows]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
