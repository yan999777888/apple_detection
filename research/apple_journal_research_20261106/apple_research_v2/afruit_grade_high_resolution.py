"""One predeclared 320px input-size comparison using the same Woo paired objective."""
import argparse
import os

import cv2
import numpy as np

import afruit_grade_crop_alignment as aligned
import afruit_grade_matched as experiment
import afruit_grade_model as grade
import afruit_grade_train as original
import afruit_visible_quality as visible


NATIVE_SIDE=320
OUT=grade.ROOT/"experiments/afruit_grade_woo320_v1"
REFERENCE=experiment.OUT/"paired/selection.json"


def rebuild_regions(split, rows):
    if split not in ("train","val"):
        raise ValueError("Only AF training and validation crop generation are permitted")
    proof=original.read(aligned.OUT/"crop_cache.json")[split]
    regions=aligned.OUT/f"{split}_regions.json"
    if grade.file_sha(regions)!=proof["region_sha"]:
        raise ValueError("Original deployment crop bounds changed")
    bounds=original.read(regions)
    if [r["sha256"] for r in rows]!=[r["sha256"] for r in bounds]:
        raise ValueError("Manifest and deployed crop record order differ")
    cache=OUT/f"{split}_320.npy"
    receipt_path=OUT/f"{split}_crop_receipt.json"
    if not receipt_path.exists():
        arr=np.zeros((len(rows),NATIVE_SIDE,NATIVE_SIDE,3),np.uint8)
        for i,(r,box) in enumerate(zip(rows,bounds,strict=True)):
            if box["crop"] is None:
                continue
            frame=visible.prepare_input(aligned.read_frame(r),aligned.SIDE)
            if list(frame.shape[:2])!=box["hw"]:
                raise ValueError("Capture-to-deployment resize differs")
            x,y,right,bottom=box["crop"]
            arr[i]=grade.rgb_square(frame[y:bottom,x:right])
        with cache.with_suffix(".tmp").open("wb") as f:
            np.save(f,arr,allow_pickle=False)
        os.replace(cache.with_suffix(".tmp"),cache)
        original.atomic(receipt_path,dict(region_sha256=grade.file_sha(regions),new_pixels_sha256=grade.file_sha(cache),
                                          valid=sum(b["crop"] is not None for b in bounds),n=len(rows),
                                          crop_resampling="source -> scene long edge 1280 -> deployed bounds -> 320px aspect-preserving RGB"))
    receipt=original.read(receipt_path)
    if receipt["new_pixels_sha256"]!=grade.file_sha(cache) or receipt["region_sha256"]!=grade.file_sha(regions):
        raise ValueError("Higher-resolution committed crop pixels or rectangle provenance changed")
    array=np.load(cache,allow_pickle=False)
    return [array[i] if b["crop"] is not None else None for i,b in enumerate(bounds)]


def run(action):
    reference=original.read(REFERENCE)
    grade.SIDE=NATIVE_SIDE
    experiment.OUT=OUT
    aligned.load_cache=rebuild_regions
    experiment.freeze()
    binding=dict(source_file_sha256=grade.file_sha(__file__),base_recipe_sha256=grade.file_sha(OUT/"protocol.json"),
                 reference_selection_sha256=grade.file_sha(REFERENCE),woo_sha256=grade.file_sha(experiment.AUX),
                 input_side=NATIVE_SIDE, old_input_side=224, initial_weights=grade.file_sha(aligned.OUT/"weights.pt"),
                 single_change="Grade model input canvas changes from 224 to 320, including the training-only severe anchor. Original fruit photos and original 1280 deployment crop bounds are used. Same architecture, optimizer, replay, data, checkpoint policy and class rule.",
                 test_policy="Only AF train and val; no reading held-out images or predictions.")
    path=OUT/"resolution_protocol.json"
    if path.exists() and original.read(path)!=binding:
        raise ValueError("320px comparison protocol changed")
    if not path.exists():
        original.atomic(path,binding)
    if action=="train":
        experiment.train("paired")
        selection=OUT/"paired/selection.json"
        result=original.read(selection)
        if result["af_draw_sha256"]!=reference["af_draw_sha256"] or result["auxiliary_draw_sha256"]!=reference["auxiliary_draw_sha256"]:
            raise ValueError("Input size comparison changed exposure identities")
        result.update(input_side=NATIVE_SIDE,resolution_protocol_sha256=grade.file_sha(path))
        original.atomic(selection,result)
        original.atomic(OUT/"comparison.json",dict(source="Woo only",side224=reference,side320=result,
                                                  passed=result["passed"],identical_draws=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","train"))
    cv2.setNumThreads(1)
    run(parser.parse_args().action)
