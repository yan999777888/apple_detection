"""Independent CPU artifact/parameter/metric audit; never trains or edits experiments."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,os,random,statistics,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'apple_research_v2'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('TEMP','TMP','TORCH_HOME','HF_HOME','MPLCONFIGDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','NUMBA_CACHE_DIR'):
 os.environ[key]=str(ROOT/'tests/_runtime')
import torch
torch.set_num_threads(4)
EXPECTED_SCRIPT='d2710b46e8c2203e1996cb7fb5bbcf96683649c9c5301a24791c7628bad33599'
EXPECTED_WEIGHTS='bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0'
issues=[];artifact_receipts={};input_receipts={}
def check(condition,message):
 if not condition: issues.append(message)
 return bool(condition)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def capture(path):
 value=sha(path);artifact_receipts[str(Path(path).resolve())]=value;return value
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def state_hash(state):
 h=hashlib.sha256()
 for name,tensor in sorted(state.items()):
  x=tensor.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(memoryview(x).cast('B'))
 return h.hexdigest()
def numeric_metrics(confusion,nll):
 supports=[sum(row) for row in confusion];n=sum(supports);per=[]
 for i,support in enumerate(supports):
  tp=confusion[i][i];pred=sum(row[i] for row in confusion);p=tp/pred if pred else 0.;r=tp/support if support else 0.;f=2*p*r/(p+r) if p+r else 0.
  per.append({'support':support,'predicted_count':pred,'precision':p,'recall':r,'F1':f})
 return {'n_ROIs':n,'unweighted_NLL':nll,'accuracy':sum(confusion[i][i] for i in range(len(confusion)))/n,'macro_F1':sum(x['F1'] for x in per)/len(per),'balanced_accuracy':sum(x['recall'] for x in per)/len(per),'per_class':per,'confusion_matrix':confusion}
def audit_dataset(name):
 directory=BASE/f'runs/{name}_full20_v3_1';suite_path=directory/'suite_report.json';suite=read(suite_path);capture(suite_path)
 cache_dir=BASE/f'features/{name}_full_v3_1';complete_path=cache_dir/'complete.json';complete=read(complete_path);capture(complete_path)
 feature_sha=capture(cache_dir/'features.pt');payload=torch.load(cache_dir/'features.pt',map_location='cpu',weights_only=True)
 check(feature_sha==complete['features_sha256']==suite['suite_identity']['cache_features_sha256'],f'{name}: feature SHA mismatch')
 identity=complete['identity'];check(identity==payload['identity']==suite['suite_identity']['identity'],f'{name}: cache/suite identity mismatch')
 check(identity['script_sha256']==EXPECTED_SCRIPT and identity['weights_sha256']==EXPECTED_WEIGHTS,f'{name}: source provenance mismatch')
 check(suite['phase']=='complete' and complete['phase']=='complete',f'{name}: not complete')
 check(suite['evaluation_role']=='development_non_independent' and suite['GT_ROI_input'] and not suite['test_used'] and not suite['independent_test_claimed'],f'{name}: invalid scope')
 check(suite['input_sha256_before']==suite['input_sha256_after']==complete['input_sha256_before']==complete['input_sha256_after'],f'{name}: inputs before/after mismatch')
 for path,digest in suite['input_sha256_before'].items():
  check(path not in input_receipts or input_receipts[path]==digest,f'{name}: conflicting input receipt');input_receipts[path]=digest
 check(complete['frozen_backbone_unchanged'] and complete['backbone_tensor_sha256_before']==complete['backbone_tensor_sha256_after'],f'{name}: backbone invariant mismatch')
 records=identity['records'];train_indices=[i for i,r in enumerate(records) if r['split']=='train'];dev_indices=[i for i,r in enumerate(records) if r['split']=='dev']
 features,labels=payload['features'],payload['labels'];names=suite['class_names'];classes=len(names)
 check(tuple(features.shape)==(len(records),2048) and features.dtype==torch.float32 and bool(torch.isfinite(features).all()),f'{name}: feature values invalid')
 check(labels.tolist()==[r['label_id'] for r in records] and payload['record_ids']==[r['record_id'] for r in records],f'{name}: cache rows/labels mismatch')
 expected_train,expected_dev,expected_steps=(1532,114,960) if name=='mendeley' else (1276,230,800)
 check((len(train_indices),len(dev_indices))==(expected_train,expected_dev),f'{name}: sample counts differ')
 counts=[sum(records[i]['label_id']==c for i in train_indices) for c in range(classes)]
 check(counts==[suite['label_counts']['train'][n] for n in names],f'{name}: train label counts differ')
 train_ids=[records[i]['record_id'] for i in train_indices];dev_ids=[records[i]['record_id'] for i in dev_indices]
 runs=[];pairs={};seen=set()
 for result in suite['results']:
  seed,arm=result['seed'],result['arm'];prefix=f'{name}/{seed}/{arm}';seen.add((seed,arm))
  report_path=Path(result['report']);report=read(report_path);check(capture(report_path)==result['report_sha256'],prefix+': report SHA mismatch')
  ck_path=Path(result['checkpoint']);ck_sha=capture(ck_path);check(ck_sha==result['checkpoint_sha256']==report['checkpoint_sha256'],prefix+': checkpoint SHA mismatch')
  ck=torch.load(ck_path,map_location='cpu',weights_only=True)
  check(report['phase']=='complete' and report['source_and_cache_verified'],prefix+': incomplete report')
  check(report['input_sha256_before']==report['input_sha256_after']==suite['input_sha256_before'],prefix+': run input map differs')
  check(ck['completed_epoch']==20 and ck['actual_optimizer_updates']==report['actual_optimizer_updates']==expected_steps,prefix+': budget mismatch')
  check(ck['recipe']==report['recipe']==suite['suite_identity']['recipe'],prefix+': recipes differ')
  check(ck['recipe']['selection']=='fixed_final_epoch_20' and not ck['recipe']['early_stopping'] and not report['development_used_for_model_selection'],prefix+': fixed-final contract differs')
  check(ck['manifest_sha256']==identity['manifest_sha256'] and ck['script_sha256']==EXPECTED_SCRIPT and ck['cache_features_sha256']==feature_sha and ck['backbone_source_weights_sha256']==EXPECTED_WEIGHTS and ck['backbone_tensor_sha256']==complete['backbone_tensor_sha256_after'],prefix+': checkpoint provenance differs')
  state=ck['head_state_dict'];actual_head_hash=state_hash(state)
  check(tuple(state['weight'].shape)==(classes,2048) and tuple(state['bias'].shape)==(classes,),prefix+': head shape differs')
  check(all(bool(torch.isfinite(v).all()) for v in state.values()),prefix+': head nonfinite')
  check(actual_head_hash==ck['head_tensor_sha256']==report['head_tensor_sha256_final'],prefix+': actual tensor hash mismatch')
  torch.manual_seed(seed);initial=torch.nn.Linear(2048,classes);initial_hash=state_hash(initial.state_dict())
  check(initial_hash==report['head_tensor_sha256_initial']==ck['run_identity']['initial_head_sha256']==result['initial_head_sha256'],prefix+': initialization differs')
  check(initial_hash!=actual_head_hash and report['head_changed'],prefix+': head did not change')
  expected_weights=None if arm=='none' else [expected_train/(classes*n) for n in counts]
  check(report['run_identity']['training_CE_weights']==ck['run_identity']['training_CE_weights']==expected_weights,prefix+': weights not from full train counts')
  optimizer=ck['optimizer_state_dict'];steps=[int(v['step'].item()) for v in optimizer['state'].values()]
  finite=all(not isinstance(v,torch.Tensor) or bool(torch.isfinite(v).all()) for entry in optimizer['state'].values() for v in entry.values())
  check(steps==report['actual_AdamW_parameter_steps']==[expected_steps,expected_steps] and finite,prefix+': optimizer state invalid')
  check(len(optimizer['param_groups'])==1 and optimizer['param_groups'][0]['lr']==.001 and optimizer['param_groups'][0]['weight_decay']==.0001,prefix+': optimizer hyperparameters differ')
  history=ck['history'];check(history==report['history'] and len(history)==20,prefix+': epoch history differs');orders=[];all_steps=[]
  for epoch,h in enumerate(history,1):
   order=list(range(expected_train));random.Random(seed+epoch).shuffle(order);order_sha=canonical([train_ids[i] for i in order]);orders.append(order_sha)
   expected_batch_sizes=[len(order[i:i+32]) for i in range(0,len(order),32)]
   check(h['epoch']==epoch and h['train_order_sha256']==order_sha and h['train_ROIs_used']==expected_train,prefix+f': epoch{epoch} sampling differs')
   check([step['n_ROIs'] for step in h['steps']]==expected_batch_sizes,prefix+f': epoch{epoch} tail batch differs')
   check(h['actual_optimizer_updates']==epoch*math.ceil(expected_train/32),prefix+f': epoch{epoch} cumulative updates differ')
   check(all(step['gradient_finite'] and step['optimizer_step_completed'] and math.isfinite(step['loss']) for step in h['steps']),prefix+': nonfinite/incomplete step log')
   check(math.isfinite(h['train_weighted_CE']) and abs(h['train_weighted_CE']-h['train_CE_numerator']/h['train_CE_denominator'])<1e-12,prefix+': training-loss aggregation differs')
   all_steps.extend(step['step'] for step in h['steps'])
  check(all_steps==list(range(1,expected_steps+1)) and orders==result['train_order_sha256'],prefix+': exact optimizer/order sequence differs')
  with torch.no_grad():
   logits=torch.nn.functional.linear(features[dev_indices],state['weight'],state['bias']);actual_labels=labels[dev_indices];probs=logits.softmax(1);pred=logits.argmax(1)
   nll=float(torch.nn.functional.cross_entropy(logits,actual_labels,reduction='none').double().sum().item())/expected_dev
  confusion=torch.bincount(actual_labels*classes+pred,minlength=classes*classes).reshape(classes,classes).tolist();metrics=numeric_metrics(confusion,nll)
  metric_delta=max(abs(metrics[k]-report['development_final'][k]) for k in ('unweighted_NLL','accuracy','macro_F1','balanced_accuracy'))
  check(metric_delta<1e-12 and confusion==report['development_final']['confusion_matrix'],prefix+': recomputed dev metrics differ')
  check(result['development']==report['development_final']==history[-1]['development'],prefix+': final dev report mismatch')
  for recomputed,recorded in zip(metrics['per_class'],report['development_final']['per_class']):
   check(all(recomputed[k]==recorded[k] for k in recomputed),prefix+': per-class metrics differ')
  prediction_path=report_path.parent/'development_predictions.json';predictions=read(prediction_path);capture(prediction_path)
  check([r['record_id'] for r in predictions]==dev_ids and [r['label_id'] for r in predictions]==actual_labels.tolist() and [r['predicted_id'] for r in predictions]==pred.tolist(),prefix+': saved predictions differ')
  probability_delta=max(abs(a-b) for row,values in zip(predictions,probs.tolist()) for a,b in zip(row['probabilities'],values))
  check(probability_delta==0,prefix+': saved probabilities differ')
  run={'seed':seed,'arm':arm,'checkpoint_sha256':ck_sha,'actual_optimizer_steps_per_parameter':steps,'head_initial_sha256':initial_hash,'head_final_sha256':actual_head_hash,'optimizer_state_finite':finite,'full_epochs':20,'ROIs_per_epoch':expected_train,'training_ROI_presentations':expected_train*20,'epoch_order_hashes_verified_from_seed':True,'actual_head_changed':initial_hash!=actual_head_hash,'development_metrics_recomputed':metrics,'maximum_metric_absolute_difference':metric_delta,'maximum_probability_absolute_difference':probability_delta,'class_weights':expected_weights}
  runs.append(run);pairs.setdefault(seed,[]).append((initial_hash,orders,metrics['macro_F1'],arm))
 check(seen=={(s,a) for s in (20261106,20261107,20261108) for a in ('none','inverse-frequency')} and len(runs)==6,f'{name}: run grid differs')
 paired=[]
 for seed,pair in pairs.items():
  check(len(pair)==2 and pair[0][0]==pair[1][0] and pair[0][1]==pair[1][1],f'{name}: paired initialization/order differs')
  values={r[3]:r[2] for r in pair};paired.append({'seed':seed,'same_initialization_and_all_epoch_orders':pair[0][0]==pair[1][0] and pair[0][1]==pair[1][1],'macro_F1_weighted_minus_none':values['inverse-frequency']-values['none']})
 aggregate={arm:{'macro_F1_values':[r['development_metrics_recomputed']['macro_F1'] for r in runs if r['arm']==arm],'macro_F1_mean':statistics.mean(r['development_metrics_recomputed']['macro_F1'] for r in runs if r['arm']==arm),'macro_F1_sample_std_across_seeds':statistics.stdev(r['development_metrics_recomputed']['macro_F1'] for r in runs if r['arm']==arm)} for arm in ('none','inverse-frequency')}
 return {'name':name,'train_ROIs':expected_train,'development_ROIs':expected_dev,'class_names':names,'counts':suite['label_counts'],'cache_features_sha256':feature_sha,'backbone_tensor_sha256':complete['backbone_tensor_sha256_after'],'current_input_files_to_verify':len(suite['input_sha256_before']),'runs':runs,'paired_comparisons':paired,'aggregate':aggregate,'all_weighted_arms_lower_macro_F1_than_same_seed_none':all(p['macro_F1_weighted_minus_none']<0 for p in paired),'scope':'GT-ROI inputs; development_non_independent; no full-network finetuning or independent test claim'}
def main():
 trainer=BASE/'train_attribute_full_v3_1.py';check(capture(trainer)==EXPECTED_SCRIPT,'Frozen trainer source changed')
 old=BASE/'train_attribute_full.py';check(capture(old)=='a4315b7a774e4d9e94259bae54f1c3a37b3debdc2fa5415f6e3a89b2ac1119d2','Original failed trainer changed')
 datasets=[audit_dataset(name) for name in ('mendeley','ssda')]
 current_input_mismatches=[]
 for path,expected in input_receipts.items():
  actual=sha(path)
  if actual!=expected:current_input_mismatches.append({'path':path,'expected':expected,'actual':actual})
 check(not current_input_mismatches,'Raw inputs no longer match recorded before/after SHA')
 weight_path=next(Path(p) for p,h in input_receipts.items() if h==EXPECTED_WEIGHTS)
 official=torch.load(weight_path,map_location='cpu',weights_only=True);body={k[len('backbone.body.'):]:v for k,v in official.items() if k.startswith('backbone.body.')}
 official_body_hash=state_hash(body);check(len(body)==265,'Official body key count differs')
 for d in datasets:check(d['backbone_tensor_sha256']==official_body_hash,d['name']+': frozen backbone hash differs from actual official initialization')
 del official,body
 feature_comparison_path=BASE/'source_audit/feature_equality_v3_vs_v3_1.json';feature_comparison=read(feature_comparison_path);capture(feature_comparison_path)
 check(feature_comparison['features']['bitwise_exact_equal'] and feature_comparison['features']['bitwise_unequal_elements']==0,'Old/new Mendeley features differ')
 artifact_mismatches=[p for p,h in artifact_receipts.items() if sha(p)!=h];check(not artifact_mismatches,'Experiment artifacts changed while audit read them')
 check(not torch.cuda.is_initialized(),'Audit unexpectedly initialized CUDA')
 result={'schema':'independent_full_attribute_v3_1_verification','completed_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if not issues else 'failed','issues':issues,'execution':{'device':'cpu','cuda_initialized':torch.cuda.is_initialized(),'torch':str(torch.__version__),'source_script':str(Path(__file__).resolve()),'source_script_sha256':sha(__file__),'training_executed':False,'original_reports_modified':False},'datasets':datasets,'raw_input_hash_verification':{'unique_files_rehashed':len(input_receipts),'mismatches':current_input_mismatches,'matches_recorded_training_and_extraction_before_after':not current_input_mismatches},'official_initialization':{'weights_path':str(weight_path),'weights_sha256':EXPECTED_WEIGHTS,'body_state_key_count':265,'body_tensor_sha256_recomputed_from_official_checkpoint':official_body_hash,'both_frozen_feature_caches_match_official_body':all(d['backbone_tensor_sha256']==official_body_hash for d in datasets)},'experiment_artifacts_unchanged_during_audit':not artifact_mismatches,'artifact_sha256':artifact_receipts,'mendeley_v3_vs_v3_1_features':feature_comparison,'limitations':['All reported classification metrics use ground-truth ROI inputs and non-independent development sets.','Mendeley semi-mature development support is only 3 ROIs; filename/hash exclusion cannot establish fruit/tree/scene independence.','SSDA date holdout lacks physical-fruit identities; labels are defect-region categories, not healthy fruit or commercial grade.','Seed standard deviations describe these three training seeds only, not statistical confidence about independent generalization.','Both arms and all seeds are reported; inverse-frequency loss weighting lowers macro-F1 in these six paired comparisons and is not promoted.']}
 target=BASE/'source_audit/full_attribute_v3_1_independent_verification.json'
 if target.exists():raise FileExistsError(target)
 tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,target)
 print(json.dumps({'status':result['status'],'issues':issues,'output':str(target),'input_files_rehashed':len(input_receipts),'CUDA_initialized':torch.cuda.is_initialized(),'datasets':[{'name':d['name'],'optimizer_steps_per_run':d['runs'][0]['actual_optimizer_steps_per_parameter'],'paired_weighting_deltas':[p['macro_F1_weighted_minus_none'] for p in d['paired_comparisons']],'aggregate':d['aggregate']} for d in datasets],'old_new_feature_bitwise_exact':feature_comparison['features']['bitwise_exact_equal']},ensure_ascii=False,indent=2))
 return 0 if not issues else 2
if __name__=='__main__':raise SystemExit(main())
