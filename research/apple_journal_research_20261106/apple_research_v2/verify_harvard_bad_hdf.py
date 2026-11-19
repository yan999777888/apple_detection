"""Confirm the four photographer-unknown Harvard Bad screenshots against historical HDF."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import cv2
import numpy as np
from PIL import Image,ImageDraw

from harvard_severe_source_review import OUT,ROOT


def features(p):
    gray=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_GRAYSCALE)
    h,w=gray.shape;scale=224/max(h,w)
    gray=cv2.resize(gray,(round(w*scale),round(h*scale)))
    kp,desc=cv2.SIFT_create(nfeatures=240,contrastThreshold=.018).detectAndCompute(gray,None)
    return np.float32([point.pt for point in kp]),desc


def main():
    cv2.setNumThreads(1)
    hdf=ROOT/"apple_research_v2/sources/additional_attribute_sources/hdf_real_only/raw/rot_defect"
    reference=sorted(hdf.glob('*'))
    assert len(reference)==1000
    with ThreadPoolExecutor(4) as pool: past=list(pool.map(features,reference))
    rows=json.loads((OUT/"bad_file_index.json").read_text(encoding="utf-8"))
    report=[];sheet=Image.new('RGB',(410,4*190),'white');draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        p=OUT/'bad_source'/row['name']
        assert hashlib.md5(p.read_bytes()).hexdigest()==row['md5']
        a,v=features(p)
        def compare(item):
            j,(b,w)=item
            if w is None or len(w)<6:return (0,0,j)
            matches=cv2.BFMatcher().knnMatch(v,w,k=2)
            kept=[m for m,n in matches if m.distance<.74*n.distance]
            if len(kept)<6:return (0,len(kept),j)
            _,ok=cv2.estimateAffinePartial2D(np.float32([a[m.queryIdx] for m in kept]),
                      np.float32([b[m.trainIdx] for m in kept]),method=cv2.RANSAC,ransacReprojThreshold=3.)
            return (int(ok.sum()) if ok is not None else 0,len(kept),j)
        with ThreadPoolExecutor(4) as pool:
            best=sorted(pool.map(compare,enumerate(past)),key=lambda x:(-x[0],-x[1],x[2]))[0]
        n,all_matches,j=best;q=reference[j]
        for col,(fp,title) in enumerate([(p,'Harvard '+row['file_id']),(q,q.name)]):
            im=Image.open(fp).convert('RGB');im.thumbnail((188,155));x=col*205;y=i*190
            sheet.paste(im,(x,y));draw.text((x,y+157),title,fill='black')
        report.append(dict(harvard_file_id=row["file_id"],harvard_name=p.name,harvard_official_md5=row["md5"],
                           hdf_path=q.relative_to(ROOT).as_posix(),hdf_sha256=hashlib.sha256(q.read_bytes()).hexdigest(),
                           geometric_inliers=n,descriptor_matches=all_matches))
    sheet.save(OUT/"all_four_confirmed_HDF.jpg",quality=95)
    (OUT/"hdf_geometric_overlap.json").write_text(json.dumps(dict(pairs=report,reference_count=1000,method="SIFT local matches with Lowe ratio .74 and partial-affine RANSAC at 224-pixel long side, 3-pixel tolerance; side-by-side texture review."),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print([(r["harvard_file_id"],r["geometric_inliers"]) for r in report])


if __name__=="__main__":main()
