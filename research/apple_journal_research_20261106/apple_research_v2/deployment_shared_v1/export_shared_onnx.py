"""Fixed640 FP32 shared-trunk dual-head export, isolated CPU only.

canonical_untrained_fixture is engineering evidence only. final40_ema requires
an immutable complete final40 checkpoint, matching SHA receipt and training report.
No training, installation, network or GPU execution occurs here.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
JOINT_DIR=ROOT/'apple_research_v2/joint'
OUTPUT_NAMES=['stage_detections','stage_prototypes','defect_detections']
OUTPUT_SHAPES=[[1,300,38],[1,32,160,160],[1,300,6]]
TASK_NAMES={'stage':{0:'immature apple',1:'semi-mature apple',2:'mature apple'},'defect':{0:'class_0',1:'class_1'}}


def require(condition,message):
    if not condition:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()

def fpath(path,within_root=False):
    p=Path(path).resolve()
    require(p.drive.upper()=='F:' and (not within_root or p.is_relative_to(ROOT)),'All artifacts must remain in F research root')
    return p

def configure_runtime(output):
    output=fpath(output,True)
    routes={'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','YOLO_CONFIG_DIR':'ultralytics','MPLCONFIGDIR':'matplotlib','TORCH_HOME':'torch','HF_HOME':'hf','XDG_CACHE_HOME':'cache','CUDA_CACHE_PATH':'cuda','TORCH_EXTENSIONS_DIR':'torch_extensions','PIP_CACHE_DIR':'pip'}
    for key,subdir in routes.items():
        path=output/'runtime'/subdir;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    os.environ.update(CUDA_VISIBLE_DEVICES='',YOLO_OFFLINE='1',YOLO_AUTOINSTALL='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    sys.dont_write_bytecode=True
    if str(JOINT_DIR) not in sys.path:sys.path.insert(0,str(JOINT_DIR))

@contextmanager
def offline():
    def refuse(*a,**k):raise RuntimeError('No network during shared model deployment preparation')
    saved=(socket.create_connection,socket.socket.connect,socket.socket.connect_ex)
    socket.create_connection=socket.socket.connect=socket.socket.connect_ex=refuse
    try:yield
    finally:socket.create_connection,socket.socket.connect,socket.socket.connect_ex=saved


def source_preflight(kind,source,expected_sha256,training_report=None):
    source=fpath(source);require(source.is_file(),'Missing local checkpoint')
    require(sha(source)==expected_sha256.lower(),'Source checkpoint checksum mismatch')
    sidecar=source.with_suffix('.sha256.json');require(sidecar.is_file(),'Source SHA receipt missing')
    receipt=read(sidecar);require(receipt['sha256']==expected_sha256.lower(),'Source sidecar SHA mismatch')
    result={'source_kind':kind,'source_checkpoint':str(source),'source_sha256':sha(source),'source_sidecar':str(sidecar),'source_sidecar_sha256':sha(sidecar),'trained':kind=='final40_ema'}
    if kind=='canonical_untrained_fixture':
        require(receipt['schema']=='joint_exact_checkpoint_sha_v1' and receipt['epoch_completed']==0 and receipt['resumable'] is False,'Fixture must be canonical epoch0 initialization')
        require(training_report is None,'A fixture cannot claim a completed training report')
    elif kind=='final40_ema':
        require(training_report is not None,'Final40 export requires complete training report')
        report_path=fpath(training_report);report=read(report_path)
        require(report.get('schema')=='joint_full40_training_report_v1' and report.get('phase')=='complete' and report.get('epochs_completed')==40,'Training is not complete at fixed40')
        require(report.get('primary_selection')=='fixed final40 EMA','Final40 EMA must be the registered primary checkpoint')
        final=report['final_checkpoint'];require(fpath(final['file'])==source and final['sha256']==expected_sha256.lower(),'Report does not identify this immutable final checkpoint')
        require(source.name=='final40_complete.pt' and receipt['epoch_completed']==40,'Interim or development-best checkpoint not permitted')
        status=report_path.parent/'status.json';require(status.is_file() and read(status).get('phase')=='complete','Training status is not complete')
        result.update(training_report=str(report_path),training_report_sha256=sha(report_path),training_status=str(status),training_status_sha256=sha(status))
    else:raise ValueError('Unknown source kind')
    result['frozen_joint_source_sha256']={str(p):sha(p) for p in sorted(JOINT_DIR.glob('*.py'))}
    return result


def load_source(kind,source,expected_sha256,training_report=None):
    """Load one real shared model; caller configures F runtime before importing torch."""
    info=source_preflight(kind,source,expected_sha256,training_report)
    import torch
    from joint_model_v1 import build_model,apply_initialization,tensor_sha,finite_state
    model=build_model('joint',seed=20261106,epochs=40,device='cpu')
    if kind=='canonical_untrained_fixture':
        apply_initialization(model,source)
        info['epoch_completed']=0
    else:
        state=torch.load(source,map_location='cpu',weights_only=False)
        require(state.get('schema')=='joint_exact_training_checkpoint_v1' and state.get('mode')=='joint' and state.get('epoch_completed')==40,'Not a final40 shared-model checkpoint')
        require(state.get('global_updates')==3480 and state.get('amp') is False,'Final training recipe/update mismatch')
        require(state['ema']['updates']==state['group_updates'],'Checkpoint EMA update audit failed')
        require(all(sha(path)==expected for path,expected in state['identity']['source_code'].items()),'Frozen training implementation differs from checkpoint identity')
        model.load_state_dict(state['ema']['model'],strict=True)
        require(tensor_sha(model.state_dict())==state['ema_tensor_sha256'],'Loaded EMA tensor checksum mismatch')
        info.update(epoch_completed=40,ema_tensor_sha256=state['ema_tensor_sha256'],checkpoint_identity_sha256=state['identity_sha256'],global_updates=state['global_updates'])
    model.criteria={}
    model.eval().requires_grad_(False)
    finite_state(model.state_dict(),'Shared export source')
    require(model.mode=='joint' and model.trunk_for('stage') is model.trunk_for('defect'),'One shared trunk is mandatory')
    require(model.names==TASK_NAMES,'Source class meanings changed')
    require(all(not p.requires_grad and p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters()),'Export must be CPU FP32')
    info['model_tensor_sha256']=tensor_sha(model.state_dict())
    info['frozen_joint_eval_sha256']=sha(JOINT_DIR/'joint_eval_v1.py')
    return model,info


def inspect_graph(path,metadata):
    import onnx
    import numpy as np
    from onnx import numpy_helper
    graph=onnx.load(path,load_external_data=True);onnx.checker.check_model(graph,full_check=True)
    def shape(v):return [int(d.dim_value) if d.HasField('dim_value') else d.dim_param for d in v.type.tensor_type.shape.dim]
    require(len(graph.graph.input)==1 and graph.graph.input[0].name=='images' and shape(graph.graph.input[0])==[1,3,640,640],'Invalid static input contract')
    require([v.name for v in graph.graph.output]==OUTPUT_NAMES,'ONNX output order/names changed')
    require([shape(v) for v in graph.graph.output]==OUTPUT_SHAPES,'ONNX output shape changed')
    require(all(v.type.tensor_type.elem_type==onnx.TensorProto.FLOAT for v in [*graph.graph.input,*graph.graph.output]),'Inputs/outputs must be FP32')
    require(all(not tensor.external_data for tensor in graph.graph.initializer),'External weight data is not allowed in this artifact')
    checked=0
    def tensor_ok(t):
        nonlocal checked
        array=numpy_helper.to_array(t)
        if np.issubdtype(array.dtype,np.floating):require(bool(np.isfinite(array).all()),'Nonfinite ONNX tensor')
        checked+=1
    for tensor in graph.graph.initializer:tensor_ok(tensor)
    for node in graph.graph.node:
        require(node.op_type not in ('NonMaxSuppression','QuantizeLinear','DequantizeLinear'),'Primary graph cannot add NMS or quantization')
        for attribute in node.attribute:
            if attribute.type==onnx.AttributeProto.TENSOR:tensor_ok(attribute.t)
    actual={p.key:p.value for p in graph.metadata_props}
    require(actual==metadata,'Graph metadata differs from export receipt')
    # One shared first convolution is direct structural evidence against tracing
    # independent backbone calls into the two branches.
    first=[n for n in graph.graph.node if n.op_type=='Conv' and n.input and n.input[0]=='images']
    require(len(first)==1,'Expected exactly one shared first convolution consuming images')
    producers={output:node for node in graph.graph.node for output in node.output}
    first_output=first[0].output[0]
    def ancestors(value):
        seen=set();pending=[value]
        while pending:
            item=pending.pop()
            if item in seen:continue
            seen.add(item)
            if item in producers:pending.extend(producers[item].input)
        return seen
    require(all(first_output in ancestors(name) for name in OUTPUT_NAMES),'An output is not connected to the same first trunk convolution')
    return {'checker_full_check':True,'input_shape':[1,3,640,640],'output_names':OUTPUT_NAMES,'output_shapes':OUTPUT_SHAPES,'finite_tensors_checked':checked,'one_input_conv':True,'all_outputs_share_first_conv':True,'node_count':len(graph.graph.node),'initializers':len(graph.graph.initializer),'metadata':actual}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-kind',choices=['canonical_untrained_fixture','final40_ema'],required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--expected-sha256',required=True)
    ap.add_argument('--training-report',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    info=source_preflight(args.source_kind,args.checkpoint,args.expected_sha256,args.training_report)
    output=fpath(args.output,True);require(not output.exists(),'Refusing overwrite/resume of export directory')
    if args.check_only:
        print(json.dumps({'phase':'preflight_passed','exported':False,**info},ensure_ascii=False));return
    output.mkdir(parents=True,exist_ok=False);configure_runtime(output)
    receipt={'schema':'shared_dual_head_export_receipt_v1','phase':'started','started_utc':now(),'source':info,'script_sha256':sha(__file__),'export':{'opset':17,'dynamo':False,'device':'cpu','quantize':32,'batch':1,'imgsz':640,'dynamic':False,'simplify':False,'extra_nms':False},'cuda_execution_proven':False,'timing_performed':False}
    try:
        import torch
        import onnx
        import ultralytics
        require(ultralytics.__version__=='8.4.115','Unexpected Ultralytics version')
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        with offline():
            model,info=load_source(args.source_kind,args.checkpoint,args.expected_sha256,args.training_report)
            receipt['source']=info
            # Isolated byte-identical source copy; the loader reads only the source,
            # and the export graph owns this new in-memory model instance.
            snapshot=output/'source_checkpoint.pt';shutil.copyfile(args.checkpoint,snapshot)
            require(sha(snapshot)==info['source_sha256'],'Checkpoint snapshot mismatch')
            class SharedOutputs(torch.nn.Module):
                def __init__(self,network):super().__init__();self.shared=network
                def forward(self,images):
                    values=self.shared.predict_both(images)
                    return values['stage'][0][0],values['stage'][0][1],values['defect'][0]
            wrapper=SharedOutputs(model).eval()
            sample=torch.zeros(1,3,640,640,dtype=torch.float32)
            calls={'trunk':0,'stage':0,'defect':0}
            def hook(key):
                def called(*_):calls[key]+=1
                return called
            handles=[model.trunk_for('stage')[0].register_forward_hook(hook('trunk'))]+[model.head_for(t).register_forward_hook(hook(t)) for t in ('stage','defect')]
            with torch.inference_mode():sample_out=wrapper(sample)
            for h in handles:h.remove()
            require(calls=={'trunk':1,'stage':1,'defect':1},'Native export wrapper executed the trunk more than once')
            require([list(x.shape) for x in sample_out]==OUTPUT_SHAPES and all(torch.isfinite(x).all().item() for x in sample_out),'Native fixed-output contract failed')
            del sample_out
            target=output/'shared_dual_head_fp32.onnx'
            with torch.no_grad():
                torch.onnx.export(wrapper,sample,str(target),opset_version=17,input_names=['images'],output_names=OUTPUT_NAMES,dynamic_axes=None,do_constant_folding=True,training=torch.onnx.TrainingMode.EVAL,export_params=True,dynamo=False,external_data=False)
            graph=onnx.load(target)
            metadata={'schema':'shared_dual_head_onnx_v1','task_names':json.dumps(TASK_NAMES,sort_keys=True),'task_types':json.dumps({'stage':'segment','defect':'detect'},sort_keys=True),'shared_trunk':'true','end2end':'true','input_shape':json.dumps([1,3,640,640]),'output_names':json.dumps(OUTPUT_NAMES),'output_shapes':json.dumps(OUTPUT_SHAPES),'source_kind':args.source_kind,'source_sha256':info['source_sha256'],'model_tensor_sha256':info['model_tensor_sha256'],'frozen_joint_eval_sha256':info['frozen_joint_eval_sha256'],'precision':'FP32','imgsz':'640','batch':'1','extra_nms':'false'}
            onnx.helper.set_model_props(graph,metadata);onnx.save(graph,target)
            inspection=inspect_graph(target,metadata)
            from joint_model_v1 import tensor_sha
            require(tensor_sha(model.state_dict())==info['model_tensor_sha256'],'Exporter changed source model values')
            require(not torch.cuda.is_initialized(),'CPU export unexpectedly initialized CUDA')
        for path,digest in info['frozen_joint_source_sha256'].items():require(sha(path)==digest,'Frozen core source changed during export')
        require(source_preflight(args.source_kind,args.checkpoint,args.expected_sha256,args.training_report)=={k:v for k,v in info.items() if k not in ('epoch_completed','ema_tensor_sha256','checkpoint_identity_sha256','global_updates','model_tensor_sha256','frozen_joint_eval_sha256')},'Source authority changed during export')
        require(sha(snapshot)==info['source_sha256'] and sha(__file__)==receipt['script_sha256'],'Snapshot/export code changed')
        license_file=ROOT/'assets/ultralytics_yolo26_coco_20261106/LICENSE.AGPL-3.0.txt'
        shutil.copyfile(license_file,output/'LICENSE.AGPL-3.0.txt')
        receipt.update(phase='complete',completed_utc=now(),onnx_path=str(target),onnx_sha256=sha(target),onnx_bytes=target.stat().st_size,onnx_inspection=inspection,native_call_counts=calls,source_unchanged=True,snapshot_sha256=sha(snapshot),license_sha256=sha(license_file),runtime={'torch':str(torch.__version__),'onnx':onnx.__version__,'ultralytics':ultralytics.__version__,'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'cuda_initialized':torch.cuda.is_initialized()})
        save(output/'export_receipt.json',receipt)
        print(json.dumps({'phase':'complete','receipt':str(output/'export_receipt.json'),'source_kind':args.source_kind,'onnx_sha256':sha(target)},ensure_ascii=False),flush=True)
    except BaseException as exc:
        receipt.update(phase='failed',error=repr(exc),traceback=traceback.format_exc());save(output/'FAILED.json',receipt);raise

if __name__=='__main__':main()
