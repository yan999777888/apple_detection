"""Count the frozen 320 grader's in-sample and held-capture decisions on actual fruit crops."""
from collections import Counter
import torch
from grade_stage_neck import OUT, grade, io


def main():
    result={}
    for split in ("train","val"):
        data=torch.load(OUT/f"{split}_features.pt",map_location="cpu",weights_only=True)
        matrix=torch.zeros((3,3),dtype=torch.int64)
        groups=Counter()
        for r,p in zip(data["records"],data["parent"].argmax(dim=1).tolist(),strict=True):
            if r["domain"]!="AF" or r["view"]!="auto":continue
            matrix[r["grade"],p]+=1
            if r["grade"]!=p:groups[(r["group"],r["grade"],p)]+=1
        result[split]=dict(resolved_auto_confusion=matrix.tolist(),
            unresolved=dict(Counter(r["grade"] for r in data["unresolved"])),
            mismatches_by_burst=[dict(group=g,truth=t,prediction=p,count=n) for (g,t,p),n in sorted(groups.items())])
    io.atomic(OUT/"frozen_parent_transfer_gap.json",dict(results=result,
        source_protocol_sha256=grade.file_sha(OUT/"protocol.json"),
        scope="Actual native320 parent outputs before the new head. Train was used in earlier model development and cannot estimate generalization. Val is a different AF capture split."))
    for partition,counts in result.items():print(partition,counts["resolved_auto_confusion"],counts["unresolved"])


if __name__=="__main__":main()
