"""Fixed descriptive examples from train-only held-out OOF outputs, unrelated to dev selection."""
import cv2
import numpy as np
from class0_patch_refinement import ROOT,OLD,read,atomic_json,load_image,manifest
from audit_defect_causes_20261112 import ious
from defect_continue_penn_diagnostic import OUT

def main():
    lut=manifest()
    best=[]
    for f in range(3):
        for r in read(OLD/f"all_mode_OOF_calibration/fold{f}_sources.json")["rows"]:
            assert lut[r["image_id"]]["partition"]=="train"
            p=np.asarray(r["sources"]["o2o"]).reshape(-1,6)
            p=p[(p[:,5]==1)&(p[:,4]>=.34)]
            ov=ious(p[:,:4],[g["box"] for g in r["GT"]])
            p=p[ov.max(1)<.1] if len(r["GT"]) else p
            if len(p):
                chosen=p[np.argmax(p[:,4])]
                best.append(dict(image_id=r["image_id"],fold=f,pred=chosen.tolist()))
    best.sort(key=lambda r:-r["pred"][4])
    examples=best[:8]
    tiles=[]
    for i,r in enumerate(examples):
        box=r["pred"]
        im=load_image(lut[r["image_id"]]["rgb_path"])
        x,y,X,Y=box[:4]
        s=round(max(94,2*max(X-x,Y-y)))
        a,b=int(np.clip((x+X)/2-s/2,0,im.shape[1]-s)),int(np.clip((y+Y)/2-s/2,0,im.shape[0]-s))
        tile=cv2.resize(im[b:b+s,a:a+s],(130,130))
        desc=np.full((20,130,3),245,np.uint8)
        cv2.putText(desc,f'OOF {i+1}  {box[4]:.2f}',(3,14),cv2.FONT_HERSHEY_SIMPLEX,.42,(15,15,15),1)
        tiles.append(np.vstack([desc,tile]))
    canvas=np.vstack([np.hstack(tiles[i:i+4]) for i in (0,4)])
    (OUT/"ssda_oof_examples.jpg").write_bytes(cv2.imencode(".jpg",canvas)[1])
    atomic_json(OUT/"ssda_oof_examples.json",dict(source="original Nano train-only held-out, diagnostic examples",
                selection="one highest-confidence insect prediction per training OOF image with IoU<.1 to all GT, eight highest",
                eligible_images=len(best),examples=examples))
    print("eligible train OOF images",len(best))
if __name__=="__main__":
    main()
