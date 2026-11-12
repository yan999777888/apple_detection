"""Compare fixed-sample combined predictions; this is not accuracy evaluation."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--historical-fused", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    fresh = json.loads(a.benchmark.read_text(encoding="utf-8-sig"))
    previous = json.loads(a.historical_fused.read_text(encoding="utf-8-sig"))
    if fresh["phase"] != "complete":
        raise ValueError("Require a complete benchmark before comparison")
    old = {x["image_id"]: x["pred"] for x in previous}
    rows = []
    for pred in fresh["cases"]["cutoff035"]["first_round_predictions"]:
        expected = old[pred["image_id"]]
        indices = [i for i, score in enumerate(expected["scores"]) if score >= .75]
        boxes = [expected["boxes"][i] for i in indices]
        scores = [expected["scores"][i] for i in indices]
        same_count = len(boxes) == len(pred["boxes"])
        rows.append(dict(image_id=pred["image_id"], fresh_count=len(pred["boxes"]),
            historical_count=len(boxes), ordered_boxes_scores_exact=boxes==pred["boxes"] and scores==pred["scores"],
            max_box_abs_diff_px=max((abs(x-y) for b,c in zip(boxes,pred["boxes"]) for x,y in zip(b,c)), default=0) if same_count else None,
            max_score_abs_diff=max((abs(x-y) for x,y in zip(scores,pred["scores"])), default=0) if same_count else None))
    report = dict(schema="combined_output_fixed_sample_comparison_v1", rows=rows,
        all_ordered_outputs_exact=all(r["ordered_boxes_scores_exact"] for r in rows),
        sources={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest()
                 for x in (a.benchmark,a.historical_fused,Path(__file__))},
        scope="8 selected development frames; .35 branch cutoff, fused output >=.75; no GT scoring",
        complete_validation_rerun=False, new_accuracy_claim=False,
        note="Ordered equality is checked without tolerance; a difference is not itself an accuracy loss.")
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__":
    main()
