"""Registered score-free arbitration between two historical grade models on development only."""
import cv2
import torch
import afruit_grade_model as grade
import afruit_grade_train as io
from grade_detail_patch import pack
from grade_severity_agreement import ARCH, SeverityAgreement
import select_grade_repair as dev


BASE=grade.ROOT/"experiments/grade_pairs_20261119"
OUT=BASE/"severity_agreement"


@torch.inference_mode()
def main():
    if (OUT/"selection.json").exists():raise FileExistsError(OUT)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    parents=io.read(BASE/"protocol.json")["reference"]
    folders={"original":grade.ROOT/"experiments/afruit_grade_release_v1",
             "native320":grade.ROOT/"experiments/grade_error49_v1/border_native320_woo_ref"}
    if any(grade.file_sha(d/"weights.pt")!=parents[n] for n,d in folders.items()):
        raise ValueError("Changed parent weights")
    io.atomic(OUT/"protocol.json",dict(parents={n:parents[n] for n in folders},
        fit="No new training, fitted scalar, threshold search or dev-label-dependent branch.",
        rule="If native320 predicts severe but original predicts non-severe, return mild. If native320 predicts good but original predicts severe, return mild. Otherwise keep native320. No severe output may become good. Both models see the actual same visible fruit; no image IDs passed to model.",
        goal="Recover mild/severe confusions and lower release risk only when independent historical graders disagree about severe. Apply common development gates."))
    model=SeverityAgreement().cpu().eval()
    for attr,folder in (("published",folders["original"]),("native",folders["native320"])):
        getattr(model,attr).load_state_dict(torch.load(folder/"weights.pt",map_location="cpu",weights_only=True))
    io.atomic(OUT/"weights.pt",model.state_dict(),tensor=True)
    panels=dev.input_panels(detail=True)
    matrices={};rows=[]
    for panel,(metadata,small,detail) in panels.items():
        labels=[None]*len(metadata)
        ids=[i for i,b in enumerate(small) if b is not None]
        for start in range(0,len(ids),8):
            chosen=ids[start:start+8]
            x=pack([small[i] for i in chosen],[detail[i] for i in chosen],"cpu")
            a=model.native(x).argmax(1).tolist()
            b=model.published(x[:,3:,:224,:224].contiguous(memory_format=torch.channels_last)).argmax(1).tolist()
            result=model(x).argmax(1).tolist()
            for i,old,pub,new in zip(chosen,a,b,result,strict=True):
                expected=1 if (old==2 and pub!=2) or (old==0 and pub==2) else old
                if expected!=new:raise ValueError("Arbitration implementation mismatch")
                labels[i]=new
                rows.append(dict(panel=panel,record=metadata[i],native320=old,published=pub,proposed=new))
        matrices[panel]=dev.counts(metadata,labels)
    x=matrices["auto"];h=matrices["whole"]
    passed=x["mild"]>=6 and x["good"]>=20 and x["severe_good"]==0 and x["severe"]>=16 and h["severe_good"]==0
    io.atomic(OUT/"development_predictions.json",dict(rows=rows,metrics=matrices))
    io.atomic(OUT/"selection.json",dict(architecture=ARCH,threshold=1.,input_side=320,passed=passed,metrics=matrices,
        weights_sha256=grade.file_sha(OUT/"weights.pt"),protocol_sha256=grade.file_sha(OUT/"protocol.json")))
    print(passed,matrices["auto"]["confusion"],matrices["whole"]["confusion"],flush=True)


if __name__=="__main__":
    main()
