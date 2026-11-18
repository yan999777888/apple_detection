"""Canonical fixed-point, error identity and timing comparison for the native-photo control."""
from argparse import Namespace
from class0_patch_refinement import read,sha,atomic_json
from defect_continue_native_partial import OUT,SNAP
from defect_continue_public_analysis import read_mode,comparison
import defect_continue_public_training as base
import compare_defect_target9485 as canonical


def main():
    ref=read_mode(base.mode_folder("replay"))
    pasted=read_mode(base.mode_folder("external",snapshot=SNAP))
    cur=read_mode(OUT)
    speed=read(OUT/"speed.json")
    assert speed["n"]==600
    canonical.run(Namespace(fixed=OUT/"fixed_predictions.json",candidates=OUT/"low_predictions.json",
                            speed=OUT/"speed.json",speed_key=None,output=OUT/"canonical_audit.json"))
    canonical_stats=read(OUT/"canonical_audit.json")["fixed_metrics"]
    accepted=sum(r["external_updates_accepted"] for r in read(OUT/"training.json"))
    out=dict(protocol=read(OUT/"protocol.json"),native_photo=cur["fixed"],classwise=canonical_stats,
             replay=ref["fixed"],crop_transfer=pasted["fixed"],replay_gt_transitions=comparison(ref,cur),
             transfer_gt_transitions=comparison(pasted,cur),false_positives=cur["false_positives"],
             accepted_updates=accepted,total_updates=400,speed=speed,weight_sha256=sha(OUT/"weights.pt"))
    atomic_json(OUT/"comparison.json",out)
    print({k:v for k,v in out.items() if k in ("native_photo","accepted_updates","speed","replay")},flush=True)
if __name__=="__main__":
    main()
