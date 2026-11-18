"""Relate each of the 29 audited misses to head decisions and corroborating detections."""

from collections import Counter
import numpy as np
from class0_patch_refinement import ROOT, atomic_json, configure, read
from defect_continue_teacher_proxy import OUT
from defect_coordinate_vote_20261118 import vote
from audit_defect_causes_20261112 import match, ious
from compare_defect_target9485 import BASELINES, PHASE, indexed, load_candidates


def main():
    src = indexed(read(PHASE))
    fixed = load_candidates(OUT/"tradeoff_predictions.json",src)
    low = load_candidates(OUT/"low_predictions.json",src)
    small = load_candidates(ROOT/"experiments/defect9485_small_v1/low_predictions.json",src)
    cross = load_candidates(ROOT/"experiments/defect9485_cross/yolo26s/one_fixed_plus_other_support_and_frozen_set_predictions.json",src)
    base = load_candidates(BASELINES["original_balanced"],src)
    catalog = read(ROOT/"experiments/class0_visual_review_v1/audit.json")["cases"]["miss"]
    assert len(catalog) == 29
    groups = Counter()
    ledger = []
    for sid,r in src.items():
        gt = r["GT"]
        prim = vote(r["sources"],.5,[.008,.008])
        new_pairs, ov = match(fixed[sid],gt)
        def support(box, pool):
            eligible = pool[pool[:,5]==box[5]]
            if not len(eligible):
                return 0.
            ov = ious([box[:4]],eligible[:,:4])[0]
            ok = ov >= .5
            return float(eligible[ok,4].max()) if ok.any() else 0.
        for i,b in enumerate(fixed[sid]):
            one = support(b,prim)
            two = support(b,small[sid])
            source = "both" if min(one,two)>=.08 else "nano" if one>=.08 else "small" if two>=.08 else "weak"
            event = "TP" if i in new_pairs else "FP"
            groups[f"class{int(b[5])}/{source}/{event}"] += 1
        small_cut = small[sid][small[sid][:,4] >= np.array([.40,.34])[small[sid][:,5].astype(int)]]
        for row in catalog:
            if row["image"] != sid:
                continue
            gi = row["index"]
            assert gt[gi]["class"] == 0 and np.allclose(gt[gi]["box"],row["box"])
            target = [*row["box"],1.,0.]
            ledger.append(dict(code=row["code"],image=sid,index=gi,previous_cause=row["category"],
                               previous_dual_recovered=row["recovered"],
                               original_crop_hit=gi in match(base[sid],gt)[0].values(),
                               small_hit=gi in match(small_cut,gt)[0].values(),
                               old_cross_hit=gi in match(cross[sid],gt)[0].values(),
                               new_fixed_hit=gi in new_pairs.values(),
                               valid_new_candidate_score=support(target,low[sid]),
                               valid_nano_vote_score=support(target,prim),
                               valid_small_score=support(target,small[sid])))
    assert len(ledger) == 29
    tally = {k:sum(r[k] for r in ledger) for k in ("previous_dual_recovered","original_crop_hit","small_hit","old_cross_hit","new_fixed_hit")}
    tally["valid_new_candidate"] = sum(r["valid_new_candidate_score"]>0 for r in ledger)
    atomic_json(OUT/"error_ledger.json",dict(original_class0_misses=sorted(ledger,key=lambda r:r["code"]),
                                             count=tally,retained_evidence_at_IoU05_score008=groups,
                                             support_rule="same-class, IoU>=0.5, branch score>=0.08 for fixed detections"))
    print(tally, dict(groups),flush=True)


if __name__ == "__main__":
    configure(OUT,"cpu")
    main()
