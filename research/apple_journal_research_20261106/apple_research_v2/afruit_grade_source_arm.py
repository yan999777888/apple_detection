"""Repeat the frozen paired-source recipe on another verified public apple source."""
import argparse
from pathlib import Path

import afruit_grade_matched as experiment
import afruit_grade_model as grade
import afruit_grade_train as original


CONTROL = experiment.OUT / "control/selection.json"


def run(source, manifest, action, keep_source=None):
    if not source.isidentifier():
        raise ValueError("Specify a single experiment source name")
    manifest=manifest.resolve(strict=True)
    manifest.relative_to((grade.ROOT/"datasets").resolve())
    previous=original.read(CONTROL)
    experiment.OUT=grade.ROOT / "experiments/afruit_grade_public_sources_v1" / source
    experiment.AUX=manifest
    if keep_source is not None:
        dataset=original.read(manifest)
        filtered=[r for r in dataset["samples"] if r.get("source")==keep_source and r["split"]=="train"]
        if {r["grade"] for r in filtered}!={0,1}:
            raise ValueError("The selected public source has no paired training grades")
        pure=dict(source_filter=keep_source,original_manifest_sha256=grade.file_sha(manifest),samples=filtered)
        view=experiment.OUT/"training_source_view.json"
        if view.exists() and original.read(view)!=pure:
            raise ValueError("Training source view changed")
        if not view.exists():
            original.atomic(view,pure)
        experiment.AUX=view
    experiment.freeze()
    binding=dict(name=source,manifest=manifest.relative_to(grade.ROOT).as_posix(),
                 manifest_sha256=grade.file_sha(manifest),applied_manifest_sha256=grade.file_sha(experiment.AUX),
                 filtered_source=keep_source,control_receipt_sha256=grade.file_sha(CONTROL),
                 wrapper_sha256=grade.file_sha(__file__),recipe_sha256=grade.file_sha(experiment.OUT/"protocol.json"),
                 rule="Exact same control and protocol as first paired comparison with one new verified source; no held-out input to training or checkpoint selection.")
    lock=experiment.OUT/"source_binding.json"
    if lock.exists() and original.read(lock)!=binding:
        raise ValueError("Additional source or control was modified after freezing")
    if not lock.exists():
        original.atomic(lock,binding)
    if action=="train":
        experiment.train("paired")
        chosen=original.read(experiment.OUT/"paired/selection.json")
        if previous["af_draw_sha256"]!=chosen["af_draw_sha256"] or chosen["public_train_manifest_sha256"]!=binding["applied_manifest_sha256"]:
            raise ValueError("The two experimental arms did not retain their paired conditions")
        original.atomic(experiment.OUT/"comparison.json",dict(control=previous,paired=chosen,
                      binding_sha256=grade.file_sha(lock),passed=chosen["passed"],source=source))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","train"))
    parser.add_argument("--source",required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--keep-source",help="Restrict to this exact source ID in a frozen mixed-source manifest")
    args=parser.parse_args()
    run(args.source,args.manifest,args.action,args.keep_source)
