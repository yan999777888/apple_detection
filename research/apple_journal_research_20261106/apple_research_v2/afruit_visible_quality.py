"""Frame-level external grade, maturity instance masks and visible defect boxes."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from class0_patch_refinement import ROOT, atomic_json, configure, load_image, read, sha
from defect_blind_review_update import SOURCE_SHA, base_network
from defect_continue_scratch_net import WARM_SHA, boxes as defect_boxes
import defect_continue_scratch_specialist as specialist
from defect9485_small import PRIMARY
from class0_crop_deploy import Inference


OUT = ROOT / "experiments" / "afruit_visible_quality_v1"
STAGES = ("immature", "semi-mature", "mature")
GRADES = ("good", "minor", "severe")
DEFECTS = ("scratch", "insect")
CROP_CONTEXT = 0.10  # Fraction of the predicted fruit box added on each side.
STAGE_WEIGHT_SHA = "f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6"


def prepare_input(image, long_edge=None):
    """Downscale the received BGR frame; predictions and overlay use the returned grid."""
    if long_edge is None:
        return image
    if long_edge < 1:
        raise ValueError("long_edge must be positive")
    h,w=image.shape[:2]
    ratio=min(1.0,long_edge/max(h,w))
    if ratio==1.0:
        return image
    return cv2.resize(image,(max(1,round(w*ratio)),max(1,round(h*ratio))),interpolation=cv2.INTER_AREA)


def crop_bounds(box, shape):
    h, w = shape[:2]
    b = np.asarray(box, dtype=float)
    if len(b) != 4 or not np.isfinite(b).all() or b[0] >= b[2] or b[1] >= b[3]:
        return None
    size = (b[2:] - b[:2]) * CROP_CONTEXT
    x, y = np.floor(b[:2] - size).astype(int)
    right, bottom = np.ceil(b[2:] + size).astype(int)
    l, t, r, d = max(0, x), max(0, y), min(w, right), min(h, bottom)
    return [int(l), int(t), int(r), int(d)] if l < r and t < d else None


def associate(rows, masks, shape):
    """Attach only when exactly one visible mask covers the defect-box center."""
    output = []
    h, w = shape[:2]
    if masks.shape[1:] != (h, w):
        raise ValueError("Expected native-image masks")
    for x, y, right, bottom, confidence, cls in rows:
        col = int(np.clip((x + right) / 2, 0, w - 1))
        row = int(np.clip((y + bottom) / 2, 0, h - 1))
        owners = np.flatnonzero(masks[:, row, col])
        output.append(dict(
            bbox=[float(v) for v in (x, y, right, bottom)], score=float(confidence),
            defect=DEFECTS[int(cls)], fruit_id=int(owners[0]) if len(owners) == 1 else None,
            association=("unique" if len(owners) == 1 else "overlap" if len(owners) else "outside"),
        ))
    return output


class QualityPipeline:
    def __init__(self):
        from afruit_grade_release import load_model, batch_preprocess, decide

        self.grade, self.grade_threshold = load_model("cuda")
        self.preprocess_grade = batch_preprocess
        self.decide_grade = decide
        # Both constructors verify the parent checkpoint hashes before loading.
        self.stage = Inference(False).worker
        self.defect = base_network()
        self.predictor = self.stage.predictors["stage"]
        if not self.predictor.args.retina_masks or self.predictor.args.conf != 0.25:
            raise ValueError("Unexpected maturity deployment working point")
        names = self.stage.model.names["stage"]
        if [names[i] for i in range(3)] != ["immature apple", "semi-mature apple", "mature apple"]:
            raise ValueError("Unexpected maturity classes")

    @torch.inference_mode()
    def classify(self, crops, hints=None):
        answers = []
        for start in range(0, len(crops), 12):
            hinted=getattr(self,"preprocess_grade_with_boxes",None)
            if hinted is not None and hints is not None:
                inp=hinted(crops[start:start+12],"cuda",hints[start:start+12])
            else:
                inp = self.preprocess_grade(crops[start:start+12], "cuda")
            logits = self.grade(inp)
            grades = self.decide_grade(logits, self.grade_threshold)
            probs = logits.float().softmax(1).cpu().tolist()
            for class_id, probabilities in zip(grades.cpu().tolist(), probs, strict=True):
                answers.append(dict(grade=GRADES[class_id], grade_id=class_id,
                                    grade_probabilities=probabilities))
        return answers

    @torch.inference_mode()
    def __call__(self, image, *, known_single_fruit=False):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("Expected nonempty HWC BGR uint8 image")
        # Preserve the prior two deployment branches and fixed postprocessing scores.
        p = defect_boxes(self.defect, image)
        p = p[p[:, 4] >= np.asarray(PRIMARY)[p[:, 5].astype(int)]]
        tensor = self.predictor.preprocess([image])
        self.predictor.batch = (["frame"], [image], [""])
        stage = self.predictor.postprocess(self.stage.model.predict_source(tensor, "stage"), tensor, [image])[0].cpu()
        rows = stage.boxes.data.numpy()
        h, w = image.shape[:2]
        masks = stage.masks.data.numpy() if stage.masks is not None else np.empty((0,h,w), dtype=np.uint8)
        if len(rows) != len(masks):
            raise ValueError("Stage boxes and masks disagree")

        fruits, crops, crop_ids = [], [], []
        for i, row in enumerate(rows):
            b = row[:4]
            region = crop_bounds(b, image.shape)
            fruits.append(dict(fruit_id=i, bbox=b.tolist(), maturity=STAGES[int(row[5])],
                               maturity_score=float(row[4]), crop=region,
                               edge_touch=bool(b[0] <= 1 or b[1] <= 1 or b[2] >= w-1 or b[3] >= h-1),
                               grade=None, grade_id=None))
            if region is not None and not known_single_fruit:
                x, y, r, bottom = region
                crop_ids.append(i)
                crops.append(image[y:bottom, x:r])
        if known_single_fruit:
            # Caller explicitly supplied a single-fruit image. This result is independent
            # of whether the original maturity model detects an apple in this domain.
            crops.append(image)
        lesions=associate(p,masks,image.shape)
        if getattr(self,"preprocess_grade_with_boxes",None) is None:
            predictions=self.classify(crops)
        else:
            from grade_pairs_20261119 import map_hints
            hints=[map_hints(lesions,offset=fruits[i]["crop"][:2],owner=i) for i in crop_ids]
            if known_single_fruit:hints.append(map_hints(lesions))
            predictions=self.classify(crops,hints)
        for i, grade in zip(crop_ids, predictions[:len(crop_ids)], strict=True):
            fruits[i].update(grade)
        if known_single_fruit and len(fruits) == 1:
            fruits[0].update(predictions[-1])
        return dict(
            fruits=fruits, defects=lesions,
            known_single_fruit_grade=predictions[-1] if known_single_fruit else None,
            grade_inference_items=len(crops), grade_batches=(len(crops)+11)//12,
            grade_scope="visible fruit appearance",
        ), masks


def overlay(image, result, masks):
    canvas = image.copy()
    colours = {"good": (65, 185, 65), "minor": (0, 185, 255), "severe": (45, 40, 230)}
    for fruit in result["fruits"]:
        contours, _ = cv2.findContours(np.ascontiguousarray(masks[fruit["fruit_id"]], dtype=np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = colours.get(fruit["grade"], (128, 128, 128))
        cv2.drawContours(canvas, contours, -1, colour, 2)
        x, y = (int(v) for v in fruit["bbox"][:2])
        cv2.putText(canvas, f'{fruit["fruit_id"]}: {fruit["grade"]} / {fruit["maturity"]}',
                    (max(1,x), max(18,y)), cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(canvas, f'{fruit["fruit_id"]}: {fruit["grade"]} / {fruit["maturity"]}',
                    (max(1,x), max(18,y)), cv2.FONT_HERSHEY_SIMPLEX,.52,colour,1,cv2.LINE_AA)
    for defect in result["defects"]:
        x,y,r,b = np.rint(defect["bbox"]).astype(int)
        cv2.rectangle(canvas,(x,y),(r,b),(255,180,70),2)
        cv2.putText(canvas,defect["defect"],(x,max(12,y-2)),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,180,70),1,cv2.LINE_AA)
    if not result["fruits"]:
        cv2.putText(canvas,"No fruit instance",(4,22),cv2.FONT_HERSHEY_SIMPLEX,.60,(200,180,50),2,cv2.LINE_AA)
    supplied = result.get("known_single_fruit_grade")
    if supplied is not None and len(result["fruits"]) != 1:
        cv2.putText(canvas, f'Supplied fruit: {supplied["grade"]}', (4,44),
                    cv2.FONT_HERSHEY_SIMPLEX,.60,colours[supplied["grade"]],2,cv2.LINE_AA)
    return canvas


def smoke():
    assert crop_bounds([20,15,40,35], (50,60)) == [18,13,42,37]
    assert crop_bounds([-1,-2,10,20], (22,11)) == [0,0,11,22]
    assert crop_bounds([8,7,8,7],(10,10)) is None
    masks = np.zeros((2, 30, 30), dtype=np.uint8)
    masks[0,1:12,1:12] = masks[1,8:20,8:20] = 1
    boxes = np.array([[1,1,3,3,.6,1], [7,7,11,11,.6,0], [20,20,23,23,.4,1]])
    results=associate(boxes,masks,(30,30))
    assert [(v["association"],v["fruit_id"]) for v in results] == [("unique",0),("overlap",None),("outside",None)]
    assert associate(boxes[:0],masks[:0],(30,30)) == []
    print("PASS: crop bounds, edge clipping, zero/multiple/overlapping instances and orphan lesions.")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--known-single-fruit", action="store_true")
    parser.add_argument("--grade-directory", type=Path, help="Optional tested grade model directory with weights.pt and selection.json")
    parser.add_argument("--long-edge", type=int, help="Maximum input long edge in pixels; coordinates and overlay refer to resized image")
    parser.add_argument("--opencv-threads", type=int, choices=(1,8), default=1,
                        help="CPU threads for decoding and resizing high-resolution frames")
    parser.add_argument("--output", type=Path, default=OUT/"preview")
    args=parser.parse_args()
    configure(args.output, "cpu" if args.smoke else "cuda")
    if args.smoke:
        smoke()
    elif args.image:
        cv2.setNumThreads(args.opencv_threads)
        torch.set_num_threads(2)
        im=cv2.imdecode(np.fromfile(args.image,dtype=np.uint8),cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"Cannot decode input image: {args.image}")
        source_hw=list(im.shape[:2])
        im=prepare_input(im,args.long_edge)
        pipe=QualityPipeline()
        if args.grade_directory:
            from matched_grade_loader import install
            side=install(pipe,args.grade_directory)
        result,masks=pipe(im,known_single_fruit=args.known_single_fruit)
        if args.grade_directory:
            result["quality_model"] = dict(sha256=sha(args.grade_directory/"weights.pt"),crop_input_side=side)
        result.update(source_hw=source_hw,prediction_hw=list(im.shape[:2]),opencv_threads=args.opencv_threads,
                      coordinate_frame="inference frame in pixels. To map x/y to the source multiply by source/inference width/height respectively.")
        atomic_json(args.output/"prediction.json",result)
        cv2.imencode(".jpg",overlay(im,result,masks))[1].tofile(str(args.output/"prediction.jpg"))
    else:
        parser.error("--image or --smoke is required")
