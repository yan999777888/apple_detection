"""One-frame entry for the retained FP32 stage-segmentation/defect models.

GPU model loading and image disk IO are intentionally outside the deployment
benchmark. Outputs are separate task predictions, not associated fruit grades.
"""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,sha,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output directory required')
    out=configure(a.output,'cuda:0')
    import cv2,torch
    from joint_model_v1 import build_model
    from deployment_adapter_v1 import FrozenDeploymentAdapter
    checkpoint=ROOT/'experiments/strong_defect_composite_20261112_v1/composite_frozen.pt'
    digest='f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6'
    if sha(checkpoint)!=digest:raise ValueError('Retained checkpoint checksum mismatch')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=build_model('separate',device='cpu');model.load_state_dict(payload['model'],strict=True);del payload
    worker=FrozenDeploymentAdapter(model.eval().requires_grad_(False).to('cuda:0'))
    im=cv2.imread(str(a.image))
    if im is None:raise ValueError('Unreadable input image')
    results=worker.predict(im,cpu=True);output={}
    for task,items in results.items():
        r=items[0];output[task]={'names':r.names,'boxes_xyxy_conf_class':r.boxes.data.tolist(),'mask_files':[]}
        if r.masks is not None:
            for i,mask in enumerate(r.masks.data.numpy()):
                file=out/f'{task}_mask_{i:03d}.png'
                if not cv2.imwrite(str(file),mask.astype('uint8')*255):raise IOError('Mask write failed')
                output[task]['mask_files'].append(file.name)
    atomic_json(out/'predictions.json',{'checkpoint_sha256':digest,'precision':'FP32','imgsz':640,'conf':.25,'max_det':300,'image':str(a.image.resolve()),'outputs':output,'warning':'Independent task outputs; no fruit association, grading or 3D localization.'})
    print(str(out/'predictions.json'))


if __name__=='__main__':main()
