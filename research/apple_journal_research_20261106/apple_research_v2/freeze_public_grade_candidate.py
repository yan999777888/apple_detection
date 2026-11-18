"""Choose a single graded-fruit candidate before opening external fruit groups."""
import argparse
from pathlib import Path

import afruit_grade_model as grade
from afruit_grade_train import atomic, read


def select(folders, output):
    if output.exists():
        raise FileExistsError("A candidate is already locked for external review")
    project=grade.ROOT
    protocol=project/"experiments/grade_matched_review_v1/field_protocol.json"
    arms=[]
    for directory in folders:
        directory=directory.resolve()
        receipt=read(directory/"selection.json")
        weight=grade.file_sha(directory/"weights.pt")
        if weight!=receipt["weights_sha256"]:
            raise ValueError("Candidate weight and receipt differ")
        metrics=receipt["metrics"]
        m=metrics["auto"]["confusion"]
        if tuple(map(sum,m))!=(25,15,17):
            raise ValueError("Different validation denominators")
        # folders in preference order: simpler sourcing and existing architecture on equal results.
        score=[m[1][1],m[2][2],m[0][0],sum(m[i][i] for i in range(3))]
        arms.append(dict(directory=directory.relative_to(project).as_posix(),weight_sha256=weight,
                         selection_sha256=grade.file_sha(directory/"selection.json"),
                         passed=receipt["passed"],validation_score=score))
    eligible=[r for r in arms if r["passed"]]
    if not eligible:
        raise ValueError("No candidate qualified for the predeclared field check")
    best=max(eligible,key=lambda x: x["validation_score"])
    atomic(output,dict(protocol_sha256=grade.file_sha(protocol),arms=arms,selected=best))
    print(best)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories",nargs="+",type=Path)
    parser.add_argument("--output",type=Path,default=grade.ROOT/"experiments/grade_matched_review_v1/frozen_candidate.json")
    args=parser.parse_args()
    select(args.directories,args.output)
