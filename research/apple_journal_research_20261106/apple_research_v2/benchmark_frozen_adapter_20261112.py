"""Full development prediction parity and counterbalanced deployment timing.

No training, downloads, precision changes or threshold changes. This tests a
dedicated immutable worker against the original guarded image-array adapter.
Official validator AP (rect=True) is a different measurement from square-input
image-array parity here; no AP is inferred from the speed experiment.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from benchmark_joint_20261112 import ROOT, sha, summary, verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        raise ValueError('Fresh output under F research root required')
    checkpoint, training = verify_run(args.run)
    if training['recipe']['mode'] != 'separate':
        raise ValueError('This bounded experiment is for the retained separate baseline')
    output.mkdir(parents=True)
    sys.path.insert(0, str(ROOT / 'apple_research_v2/joint'))
    from joint_state_v1 import configure, atomic_json, load_checkpoint
    configure(output, 'cuda:0')
    import cv2
    import torch
    from joint_model_v1 import build_model, tensor_sha
    from joint_eval_v1 import JointInferenceAdapter, source_spec, offline_network
    from deployment_adapter_v1 import FrozenDeploymentAdapter
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    state = load_checkpoint(checkpoint)
    model = build_model('separate', device='cpu')
    model.load_state_dict(state['ema']['model'], strict=True)
    before = tensor_sha(model.state_dict())
    if before != state['ema_tensor_sha256']:
        raise ValueError('EMA reconstruction mismatch')
    del state
    model.eval().requires_grad_(False).to('cuda:0')
    reference = JointInferenceAdapter(model, conf=.25, max_det=300, audit_values=False)
    fast = FrozenDeploymentAdapter(model, conf=.25, max_det=300, audit_values=False)
    sources = {task: source_spec(task)['images'] for task in ('stage', 'defect')}
    protocol = dict(
        phase='running', checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        precision='FP32', imgsz=640, rect=False, batch=1, conf=.25, max_det=300,
        scope='BGR image array to two-task CPU Results including native masks; no camera/depth/robot',
        parity_scope='185 development images, both heads, exact boxes/scores/classes and native mask pixels',
        timing_images_per_source=20, warmup=10, repeats=4,
        order='AB on even rounds, BA on odd rounds, per image; A=reference, B=frozen',
        excluded='disk decode, load, parity checks, hashing, acquisition, association, decision, depth and robot',
        script_sha256=sha(__file__), adapter_sha256=sha(Path(__file__).parent/'joint/deployment_adapter_v1.py'),
        gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
        metric_warning='Official AP uses its original rect validator. No new AP or whole-fruit grade accuracy is inferred here.')
    atomic_json(output/'protocol.json', protocol)
    pairs = []
    try:
        with offline_network(), torch.inference_mode():
            for task, paths in sources.items():
                for path in paths:
                    im = cv2.imread(path)
                    if im is None:
                        raise ValueError('Cannot decode image: '+path)
                    original = reference.predict(im, paths=[path], cpu=True)
                    deployed = fast.predict(im, paths=[path], cpu=True)
                    row = {'source':task, 'path':path, 'image_sha256':sha(path), 'heads':{}}
                    for head in ('stage', 'defect'):
                        a, b = original[head][0], deployed[head][0]
                        boxes_equal = torch.equal(a.boxes.data, b.boxes.data)
                        masks_equal = ((a.masks is None and b.masks is None) or
                                       (a.masks is not None and b.masks is not None and
                                        torch.equal(a.masks.data, b.masks.data)))
                        row['heads'][head] = {'detections':len(a.boxes),
                            'boxes_scores_classes_exact':boxes_equal, 'mask_pixels_exact':masks_equal}
                        if not boxes_equal or not masks_equal:
                            atomic_json(output/'parity_failure.json', row)
                            raise AssertionError('Prediction parity failed; no speed promotion')
                    pairs.append(row)
                    del original, deployed
            atomic_json(output/'parity.json', {'passed':True, 'images':len(pairs), 'rows':pairs})
            selected = []
            for task, paths in sources.items():
                selected.extend(paths[round(i*(len(paths)-1)/19)] for i in range(20))
            images = [cv2.imread(path) for path in selected]
            adapters = {'reference':reference, 'frozen':fast}
            for index in range(10):
                for adapter in adapters.values():
                    adapter.predict(images[index % len(images)], cpu=True)
            observations = []
            for repeat in range(4):
                order = ('reference','frozen') if repeat % 2 == 0 else ('frozen','reference')
                for path, im in zip(selected, images):
                    for name in order:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        predictions = adapters[name].predict(im, paths=[path], cpu=True)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter()-start)*1000
                        observations.append({'repeat':repeat,'path':path,'mode':name,'ms':elapsed,
                                             'detections':{h:len(predictions[h][0].boxes) for h in predictions}})
                        del predictions
            after = tensor_sha(model.state_dict())
            if before != after or sha(checkpoint) != protocol['checkpoint_sha256']:
                raise AssertionError('Model state or on-disk checkpoint changed')
            result = {**protocol, 'phase':'complete', 'parity_passed':True, 'parity_images':len(pairs),
                      'state_sha256_before':before, 'state_sha256_after':after,
                      'timing':{name:summary([x['ms'] for x in observations if x['mode']==name])
                                for name in adapters}, 'observations':observations}
            atomic_json(output/'results.json', result)
            print(json.dumps({'phase':'complete','parity_images':len(pairs),'timing':result['timing']},indent=2))
    except BaseException as exc:
        atomic_json(output/'FAILED.json', {'phase':'failed','exception':repr(exc),'parity_completed_images':len(pairs)})
        raise


if __name__ == '__main__':
    main()
