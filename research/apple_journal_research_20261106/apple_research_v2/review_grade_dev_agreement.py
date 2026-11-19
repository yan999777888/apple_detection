"""List persistent development misses across the frozen parents and completed patch experiments."""
from collections import Counter

from afruit_grade_train import atomic, read, ROOT


OUT=ROOT/"experiments/grade_pairs_20261119"
ARMS=("sparse_verifier_v1","grounded_sparse_v2","fruit_surface_burden_v1")


def main():
    def load(name):
        d=read(OUT/name/"development_predictions.json")
        rows=[r for r in d["rows"] if r["panel"]=="auto"]
        held=[r for r in d["unresolved"] if r["panel"]=="auto"]
        merged={x["sha256"]:dict(path=x["path"],truth=x["grade"],woo=x["woo"],
                                 native320=x["native320"],result=x["proposed"]) for x in rows}
        for r in held:
            merged[r["sha256"]]=dict(path=r["path"],truth=r["grade"],woo=None,native320=None,result=None)
        if len(merged)!=57:raise ValueError("Development split changed")
        return merged
    tables={arm:load(arm) for arm in ARMS}
    first=tables[ARMS[0]]
    rows=[]
    for key,r in first.items():
        result=dict(sha256=key, path=r["path"], truth=r["truth"], woo=r["woo"],native320=r["native320"])
        for arm,table in tables.items():
            q=table[key]
            if any(q[c]!=r[c] for c in ("truth","woo","native320","path")):
                raise ValueError("Reference disagreement")
            result[arm]=q["result"]
        pred=[result[k] for k in ("woo","native320",*ARMS)]
        result["status"]=("unresolved" if all(p is None for p in pred) else
                          "correct_in_all" if all(p==r["truth"] for p in pred) else
                          "wrong_in_all" if all(p!=r["truth"] for p in pred) else "mixed")
        rows.append(result)
    counts={g:dict(Counter(r["status"] for r in rows if r["truth"]==g)) for g in range(3)}
    atomic(OUT/"development_agreement.json",dict(models=["woo","native320",*ARMS],counts=counts,rows=rows,
         scope="Existing automatic development predictions, same predicted fruit boxes. Describes disagreement only, not a selectable oracle classifier or independent-validation evidence."))
    print(counts)


if __name__=="__main__":
    main()
