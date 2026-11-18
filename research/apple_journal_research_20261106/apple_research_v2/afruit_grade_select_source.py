"""Lock the choice across same-architecture training arms using only the 57 validation images."""
from pathlib import Path

import afruit_grade_matched as matched
import afruit_grade_model as grade
import afruit_grade_train as original


BASE=grade.ROOT/"experiments"
SOURCES={
    "AF_replay":BASE/"afruit_grade_matched_v1/control",
    "Woo_224":BASE/"afruit_grade_matched_v1/paired",
    "Woo_320":BASE/"afruit_grade_woo320_v1/paired",
    "FruitLife_224":BASE/"afruit_grade_public_sources_v1/fruitlife/paired",
    "Woo_FruitLife_224":BASE/"afruit_grade_public_sources_v1/fruitlife_woo/paired",
}


def lock():
    rows=[]
    draws=set()
    for arm,folder in SOURCES.items():
        selection=original.read(folder/"selection.json")
        if selection["weights_sha256"]!=grade.file_sha(folder/"weights.pt"):
            raise ValueError("Weights differ from validation selection")
        if selection["checks"]!=matched.check(selection["metrics"]):
            raise ValueError("Validation requirements changed")
        draws.add(selection["af_draw_sha256"])
        rows.append(dict(arm=arm,weights=(folder/"weights.pt").relative_to(grade.ROOT).as_posix(),
                         sha256=selection["weights_sha256"],input_side=selection.get("input_side",224),
                         epoch=selection["epoch"],passed=selection["passed"],metrics=selection["metrics"],
                         selection_sha256=grade.file_sha(folder/"selection.json")))
    if len(draws)!=1:
        raise ValueError("The five arms used different AF replay")
    eligible=[r for r in rows if r["passed"]]
    best=max(eligible,key=lambda r:matched.order(r["metrics"])) if eligible else None
    result=dict(winner=best["arm"] if best else None,weights=best["weights"] if best else None,
                sha256=best["sha256"] if best else None,input_side=best["input_side"] if best else None,
                selection_scope="57 AF development validation images only. Architecture unchanged; this is a candidate for frozen external verification.",
                rank="Same validation gates and rank function used for every source and resolution",af_draw_sha256=next(iter(draws)),arms=rows)
    output=BASE/"afruit_grade_matched_v1/source_selection.json"
    if output.exists() and original.read(output)!=result:
        raise ValueError("Selection already frozen on different inputs")
    if not output.exists():
        original.atomic(output,result)
    print(result["winner"],result["sha256"])


if __name__=="__main__":
    lock()
