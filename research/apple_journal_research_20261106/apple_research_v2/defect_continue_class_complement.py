"""Retrospective diagnostic: pair a fixed insect model with archived scratch outputs."""

from collections import defaultdict

from compare_defect_target9485 import PHASE, ROOT, indexed, load_candidates, match, read
from joint_state_v1 import atomic_json


def scratch_curve(predictions, canonical, insect):
    bins = defaultdict(lambda: [0, 0])
    for key, values in predictions.items():
        matched, _ = match(values, canonical[key]["GT"])
        for i, candidate in enumerate(values):
            if int(candidate[5]) == 0:
                bins[float(candidate[4])][int(i not in matched)] += 1
    tp, fp = insect["TP"], insect["FP"]
    points = []
    for score, (correct, incorrect) in [
        (None, (0, 0)),
        *sorted(bins.items(), reverse=True),
    ]:
        tp += correct
        fp += incorrect
        points.append(
            dict(
                TP=tp,
                FP=fp,
                FN=230 - tp,
                P=tp / (tp + fp),
                R=tp / 230,
                scratch_TP=tp - insect["TP"],
                scratch_FP=fp - insect["FP"],
                scratch_threshold=score,
            )
        )
    return dict(
        P_at_R85=max(
            (x for x in points if x["R"] >= 0.85),
            key=lambda x: (x["P"], x["R"]),
            default=None,
        ),
        R_at_P94=max(
            (x for x in points if x["P"] >= 0.94),
            key=lambda x: (x["R"], x["P"]),
            default=None,
        ),
    )


def main():
    base = ROOT / "experiments"
    dest = base / "defect_continue_warm_rank"
    canonical = indexed(read(PHASE))
    assert sum(len(row["GT"]) for row in canonical.values()) == 230
    insect = read(dest / "comparison_vs_replay.json")["fixed"]["class/1"]
    rows = []
    skipped = {}
    for path in sorted(base.glob("defect*/**/*low_predictions.json")):
        label = path.relative_to(ROOT).as_posix()
        entries = read(path)
        if not isinstance(entries, list) or not entries or "pred" not in entries[0]:
            skipped[label] = "Different saved prediction schema"
            continue
        if {r["image_id"] for r in entries} != canonical.keys():
            skipped[label] = "Different image set"
            continue
        archive = load_candidates(path, canonical)
        rows.append(dict(source=label, **scratch_curve(archive, canonical, insect)))
    out = dict(
        scope="Retrospective class-specialist diagnostic on 113 development frames. Original warm-rank class-1 threshold is fixed. Each eligible archived class-0 candidate stream is swept independently. No ensemble speed measurement or deployment selection.",
        fixed_insect=insect,
        rows=rows,
        skipped=skipped,
    )
    atomic_json(dest / "archived_scratch_complement.json", out)
    print("sources", len(rows), "skipped", len(skipped))
    for key, metric in (("P_at_R85", "P"), ("R_at_P94", "R")):
        print(key, max((r for r in rows if r[key]), key=lambda r: r[key][metric]))


if __name__ == "__main__":
    main()
