"""CPU contract tests execute the installed Model.val function, with a fake validator.

No torch import, GPU, dataset access, inference or files written during tests.
The actual val factory/callback/return control flow is preserved, including its
local-only validator variable that caused the original integration failure.
"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'apple_research_v2/validate_resolution_pairs_official.py'
spec = importlib.util.spec_from_file_location('official_validation', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
SOURCE = Path('F:/anaconda/envs/falldet/Lib/site-packages/ultralytics/engine/model.py')
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Model')
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'val')
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
namespace = {}
exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
ACTUAL_VAL = namespace['val']


class FakeModel:
    val = ACTUAL_VAL

    def __init__(self, mode='ok'):
        self.mode = mode
        self.overrides = {}
        self.callbacks = {'on_val_start': [], 'on_val_end': []}
        self.model = object()
        self.instances = []

    def __getattr__(self, key):
        if key == 'validator':
            raise AssertionError('Model.val does not expose model.validator')
        raise AttributeError(key)

    def add_callback(self, event, callback):
        self.callbacks[event].append(callback)

    def _smart_load(self, name):
        assert name == 'validator'
        owner = self
        class Validator:
            def __init__(self, args, _callbacks):
                self.args = SimpleNamespace(**args)
                self.callbacks = _callbacks
                self.device = 'cuda:0'
                self.save_dir = Path(args['project']) / args['name']
                self.metrics = SimpleNamespace(results_dict={'metrics/mAP50(B)': .75, 'metrics/mAP50-95(B)': .5}, nt_per_class=[65,3,46])
                owner.instances.append(self)

            def __call__(self, model):
                assert model is owner.model
                self.args.quantize = 16 if owner.mode == 'fp16' else None
                if owner.mode == 'raise':
                    raise RuntimeError('synthetic validation interruption')
                if owner.mode != 'missing_callbacks':
                    for fn in self.callbacks['on_val_start']:
                        fn(self)
                self.seen = 71 if owner.mode == 'coverage' else 72
                self.end2end = True
                if owner.mode == 'support':
                    self.metrics.nt_per_class = [65,4,45]
                if owner.mode == 'nonfinite':
                    self.metrics.results_dict['metrics/mAP50(B)'] = float('nan')
                if owner.mode != 'missing_callbacks':
                    for fn in self.callbacks['on_val_end']:
                        fn(self)
        return Validator


def run(model, size=640):
    return m.validate_one(model, data=ROOT/'apple_research_v2/datasets/mendeley_stage_seg_v1/data.yaml', output=ROOT/'experiments/unused_fixture_output', size=size, device='cuda:0', max_det=300, expected_images=72, expected_support=[65,3,46])


class ActualModelValTests(unittest.TestCase):
    def test_actual_local_validator_factory_both_sizes(self):
        model = FakeModel()
        for size in (640,960):
            result = run(model, size)
            self.assertEqual(result['resolution'], size)
            self.assertEqual(result['config']['effective_quantize'], None)
            self.assertTrue(result['config']['effective_rect'])
            self.assertTrue(result['callback_capture_verified'])
            self.assertEqual(result['class_support'], [65,3,46])
            self.assertEqual(result['seen'],72)
            self.assertTrue(result['end2end_head'])
            self.assertFalse(model.callbacks['on_val_start'])
            self.assertFalse(model.callbacks['on_val_end'])
        self.assertEqual(len(model.instances),2)

    def test_missing_callbacks_fail(self):
        with self.assertRaisesRegex(ValueError,'callbacks did not complete'):
            run(FakeModel('missing_callbacks'))

    def test_nonfinite_metrics_fail(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            run(FakeModel('nonfinite'))

    def test_fp16_fails(self):
        with self.assertRaisesRegex(ValueError,'non-FP32'):
            run(FakeModel('fp16'))

    def test_coverage_fails(self):
        with self.assertRaisesRegex(ValueError,'coverage differs'):
            run(FakeModel('coverage'))

    def test_support_fails(self):
        with self.assertRaisesRegex(ValueError,'class support differs'):
            run(FakeModel('support'))

    def test_failure_restores_only_our_callbacks(self):
        model=FakeModel('raise')
        existing=lambda validator: None
        model.add_callback('on_val_start',existing)
        with self.assertRaisesRegex(RuntimeError,'interruption'):
            run(model)
        self.assertEqual(model.callbacks['on_val_start'],[existing])
        self.assertEqual(model.callbacks['on_val_end'],[])


if __name__ == '__main__':
    unittest.main()