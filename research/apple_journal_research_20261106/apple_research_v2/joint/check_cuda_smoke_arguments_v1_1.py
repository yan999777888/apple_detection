"""Stdlib AST argument regression only; never imports Torch or runs a model."""
import ast,copy,hashlib,inspect,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=Path(__file__).with_name('smoke_joint_cuda_v1_1.py')
validator=Path(__file__).with_name('joint_eval_v1.py')
old=Path(__file__).with_name('smoke_joint_cuda_v1.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old)=='99cddc670226dc6b9e17ac336f8457f44c37ff1fcf99c4bffc05883bc445e64f'
assert sha(validator)=='7296fa13005421993f5f9c67b784f854c42ac6811126d245c20b580de6da5452'
a=ast.parse(source.read_text(encoding='utf-8-sig'));v=ast.parse(validator.read_text(encoding='utf-8-sig'))
function=next(node for node in v.body if isinstance(node,ast.FunctionDef) and node.name=='evaluate_task')
stub=copy.deepcopy(function);stub.body=[ast.Pass()];stub.decorator_list=[];module=ast.fix_missing_locations(ast.Module(body=[stub],type_ignores=[]));namespace={}
exec(compile(module,'signature_only','exec'),namespace)
signature=inspect.signature(namespace['evaluate_task'])
calls=[node for node in ast.walk(a) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='evaluate_task']
assert len(calls)==1
call=calls[0];constant_kwargs={kw.arg:kw.value.value for kw in call.keywords if isinstance(kw.value,ast.Constant)}
assert constant_kwargs=={'batch':1,'imgsz':640,'scope':'full_development'}
assert len(call.args)==4 and {kw.arg for kw in call.keywords}=={'device','batch','imgsz','scope'}
signature.bind(object(),'stage','actual_source.yaml','new_output',device='cuda:0',**constant_kwargs)
supported={node.comparators[0].value for node in ast.walk(function) if isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='scope' and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant)}
assert constant_kwargs['scope'] in supported
receipt_keys=set()
for node in ast.walk(function):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='receipt' and node.func.attr=='update':receipt_keys.update(kw.arg for kw in node.keywords)
    if isinstance(node,ast.Dict):receipt_keys.update(key.value for key in node.keys if isinstance(key,ast.Constant) and isinstance(key.value,str))
    if isinstance(node,ast.Assign):
        for target in node.targets:
            if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='receipt' and isinstance(target.slice,ast.Constant):receipt_keys.add(target.slice.value)
used={node.slice.value for node in ast.walk(a) if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id=='result' and isinstance(node.slice,ast.Constant)}
assert used=={'phase','seen','state_guard','scope'} and used.issubset(receipt_keys)
assert 'DataLoader' not in source.read_text() and 'get_dataloader' not in source.read_text()
report={'schema':'cuda_smoke_v1_1_stdlib_argument_contract','phase':'passed','model_execution':False,'torch_imported':False,'GPU_used':False,'signature':str(signature),'supported_scope':'full_development','full_development_expected_images':{'stage':72,'defect':113},'receipt_keys_consumed':sorted(used),'old_failed_script_preserved_sha256':sha(old),'frozen_validator_sha256':sha(validator),'new_smoke_sha256':sha(source)}
output=ROOT/'experiments/joint_training_v1/cuda_smoke_v1_1_argument_contract.json'
assert not output.exists();output.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
