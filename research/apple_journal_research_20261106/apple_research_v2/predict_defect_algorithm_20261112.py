"""Opt-in deployment entry. Baseline is default; no automatic checkpoint replacement."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
from joint_state_v1 import configure,sha,atomic_json
def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',default='o2o_nms');p.add_argument('--thresholds',type=float,nargs=2,default=[.25,.25]);p.add_argument('--checkpoint',type=Path,default=ROOT/'experiments/strong_defect_composite_20261112_v1/composite_frozen.pt');p.add_argument('--checkpoint-sha256',default='f8becdeff0e042fa19d681141aa56e133393130b4a2af23133706fd7766367f6');a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    if a.image.resolve().drive.upper()!='F:' or a.checkpoint.resolve().drive.upper()!='F:':raise ValueError('F drive only')
    configure(a.output,'cuda:0')
    import torch,cv2
    from joint_model_v1 import build_model
    from defect_algorithm_modes_20261112 import AlgorithmAdapter,MODES
    assert a.mode in MODES and all(.001<=x<=1 for x in a.thresholds)
    if sha(a.checkpoint)!=a.checkpoint_sha256:raise ValueError('Checkpoint checksum mismatch')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model=build_model('separate',device='cpu');model.load_state_dict(state['model']);del state
    im=cv2.imread(str(a.image));assert im is not None
    worker=AlgorithmAdapter(model.eval().requires_grad_(False).to('cuda:0'));results=worker.predict_mode(im,a.mode,a.thresholds);output={}
    for task,rr in results.items():
        r=rr[0];output[task]={'names':r.names,'boxes_xyxy_conf_class':r.boxes.data.tolist(),'box_coordinates':'original input image pixels','mask_files':[],'mask_tensor_hw':None}
        if r.masks is not None:
            output[task]['mask_tensor_hw']=list(r.masks.data.shape[-2:]);output[task]['mask_grid_note']='Native predictor mask tensor grid; use polygons_xy_original for original-image coordinates, not raw PNG indices.'
            output[task]['polygons_xy_original']=[p.tolist() for p in r.masks.xy]
            for i,mask in enumerate(r.masks.data.numpy()):
                path=a.output/f'{task}_mask_{i:03d}.png';assert cv2.imwrite(str(path),mask.astype('uint8')*255);output[task]['mask_files'].append(path.name)
        assert cv2.imwrite(str(a.output/f'{task}_overlay.jpg'),r.plot())
    atomic_json(a.output/'predictions.json',{'mode':a.mode,'thresholds':a.thresholds,'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha(a.checkpoint),'image':str(a.image.resolve()),'original_image_hw':list(im.shape[:2]),'outputs':output,'precision':'FP32','imgsz':640,'scope':'independent stage masks and defect boxes; no fruit association, defect segmentation or 3D claim'})
if __name__=='__main__':main()
