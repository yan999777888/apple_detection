"""CPU-only contracts for the v3.1 frozen-feature attribute experiment.

Synthetic values and images are disposable unit fixtures, never research data.
Tests exercise real shard finalization, CPU head training, and CPU extraction
with the local official COCO backbone. No research cache/head checkpoint is
read or changed, no research report is written, and no CUDA device is used.
"""
from __future__ import annotations
import ast
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
RESEARCH_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = RESEARCH_ROOT / "apple_research_v2" / "train_attribute_full_v3_1.py"
spec = importlib.util.spec_from_file_location("attribute_full_v3_1_contract_subject", TRAINER_PATH)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
SOURCE_TREE = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
FIXTURE_ROOT = RESEARCH_ROOT / "tests" / "__metadata_only_no_files__"
NAMES = ["fixture_surface_a", "fixture_surface_b"]


def fixture_document():
    rows = []
    for split in ("train", "dev"):
        for index, label in enumerate(NAMES):
            identity = f"{split}_{index}"
            rows.append({"image_path": identity + ".png", "image_sha256": "a" * 64,
                "annotation_path": identity + ".json", "annotation_sha256": "b" * 64,
                "bbox_xyxy": [1, 2, 11, 12], "label": label,
                "source_image_id": identity, "group_id": identity, "split": split})
    return {"label_semantics": "Unit fixture surface attributes only; not research labels.",
            "class_names": NAMES.copy(),
            "split_policy": {"evaluation_role": subject.ROLE,
                             "independent_group_holdout": False}, "records": rows}


def function_node(name):
    return next(node for node in SOURCE_TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)


def epoch_loop_node():
    return next(node for node in ast.walk(function_node("train"))
                if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == "epoch")


def production_epoch_indices(n, seed, epoch):
    """Execute only actual production index-generation/slicing AST, never training."""
    outer = epoch_loop_node()
    order_statement = next(node for node in outer.body if isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "order" for t in node.targets))
    shuffle_statement = next(node for node in outer.body if isinstance(node, ast.Expr)
                             and isinstance(node.value, ast.Call)
                             and isinstance(node.value.func, ast.Attribute)
                             and node.value.func.attr == "shuffle")
    inner = next(node for node in outer.body if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == "offset")
    indices_statement = next(node for node in inner.body if isinstance(node, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == "indices" for t in node.targets))
    safe_inner = copy.deepcopy(inner)
    safe_inner.body = [copy.deepcopy(indices_statement), ast.parse("batches.append(indices.copy())").body[0]]
    safe_inner.orelse = []
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(order_statement),
                   copy.deepcopy(shuffle_statement), safe_inner], type_ignores=[]))
    namespace = {"ty": range(n), "seed": seed, "epoch": epoch, "random": random,
                 "RECIPE": subject.RECIPE, "batches": []}
    exec(compile(module, str(TRAINER_PATH) + "::<indices-only>", "exec"), namespace)
    return namespace["batches"]


class ManifestContracts(unittest.TestCase):
    def normalized(self, document=None, allow=True):
        return subject.manifest_metadata(document or fixture_document(), FIXTURE_ROOT, allow)

    def test_valid_manifest_preserves_semantics_and_normalizes_val(self):
        document = fixture_document()
        document["records"][2]["split"] = "val"
        before = copy.deepcopy(document)
        rows, names = self.normalized(document)
        self.assertEqual(names, NAMES)
        self.assertEqual([r["split"] for r in rows], ["train", "train", "dev", "dev"])
        self.assertEqual([r["label_id"] for r in rows], [0, 1, 0, 1])
        self.assertEqual(len({r["record_id"] for r in rows}), 4)
        self.assertEqual(document, before)

    def test_explicit_nonindependent_development_permission_required(self):
        with self.assertRaisesRegex(ValueError, "Explicit.*required"):
            self.normalized(allow=False)

    def test_cannot_claim_independent_group_holdout(self):
        document = fixture_document()
        document["split_policy"]["independent_group_holdout"] = True
        with self.assertRaisesRegex(ValueError, "Cannot claim independent"):
            self.normalized(document)

    def test_evaluation_role_must_explicitly_be_nonindependent_development(self):
        for role in (None, "independent_test", "development_independent", "train"):
            with self.subTest(role=role):
                document = fixture_document()
                document["split_policy"]["evaluation_role"] = role
                with self.assertRaisesRegex(ValueError, "evaluation_role"):
                    self.normalized(document)

    def test_test_unknown_or_missing_split_refused_before_source_file_reads(self):
        for split in ("test", "unknown", None):
            with self.subTest(split=split):
                document = fixture_document()
                document["records"][0]["split"] = split
                args = SimpleNamespace(manifest=FIXTURE_ROOT / "manifest.json",
                    weights=FIXTURE_ROOT / "must_never_open_weights.pth",
                    allow_non_independent_development=True)
                with mock.patch.object(Path, "read_text", return_value=json.dumps(document)), \
                     mock.patch.object(Path, "open", side_effect=AssertionError("Source file was opened")) as opened, \
                     mock.patch.object(subject, "sha", side_effect=AssertionError("Source hashing started")) as hashed:
                    with self.assertRaisesRegex(ValueError, "Test/unknown split refused"):
                        subject.preflight(args)
                    opened.assert_not_called()
                    hashed.assert_not_called()

    def test_missing_semantics_annotation_or_digest_refused(self):
        document = fixture_document()
        document.pop("label_semantics")
        with self.assertRaisesRegex(ValueError, "label_semantics"):
            self.normalized(document)
        for field in ("annotation_path", "annotation_sha256", "image_sha256", "group_id"):
            with self.subTest(field=field):
                document = fixture_document()
                document["records"][0].pop(field)
                with self.assertRaisesRegex(ValueError, "missing"):
                    self.normalized(document)

    def test_source_image_and_group_cannot_cross_splits(self):
        for field in ("source_image_id", "group_id"):
            with self.subTest(field=field):
                document = fixture_document()
                document["records"][2][field] = document["records"][0][field]
                with self.assertRaisesRegex(ValueError, field + " crosses split"):
                    self.normalized(document)

    def test_image_path_cannot_cross_splits_even_with_different_roi(self):
        document = fixture_document()
        document["records"][2]["image_path"] = document["records"][0]["image_path"]
        document["records"][2]["bbox_xyxy"] = [2, 3, 12, 13]
        with self.assertRaisesRegex(ValueError, "image_path crosses split"):
            self.normalized(document)

    def test_duplicate_roi_rejected_even_with_conflicting_label(self):
        for conflicting in (False, True):
            with self.subTest(conflicting=conflicting):
                document = fixture_document()
                repeated = copy.deepcopy(document["records"][0])
                if conflicting:
                    repeated["label"] = NAMES[1]
                document["records"].append(repeated)
                with self.assertRaisesRegex(ValueError, "Duplicate ROI"):
                    self.normalized(document)

    def test_bbox_and_unknown_label_contracts(self):
        for box in ([0, 0, 0, 1], [-1, 0, 2, 2], [0, 0, float("nan"), 2], [False, 0, 2, 2]):
            with self.subTest(box=box):
                document = fixture_document()
                document["records"][0]["bbox_xyxy"] = box
                with self.assertRaisesRegex(ValueError, "bbox"):
                    self.normalized(document)
        document = fixture_document()
        document["records"][0]["label"] = "not_declared"
        with self.assertRaisesRegex(ValueError, "Unknown label"):
            self.normalized(document)

    def test_all_classes_need_real_train_and_dev_support(self):
        for missing_index, split in ((0, "train"), (2, "dev")):
            with self.subTest(split=split):
                document = fixture_document()
                document["records"].pop(missing_index)
                with self.assertRaisesRegex(ValueError, "real " + split + " support"):
                    self.normalized(document)


class WeightAndMetricContracts(unittest.TestCase):
    def test_inverse_frequency_formula_from_train_counts(self):
        self.assertEqual(subject.class_weights({"a": 3, "b": 1}, ["a", "b"], "inverse-frequency"), [2/3, 2])
        self.assertEqual(subject.class_weights({"a": 12, "b": 6, "c": 2}, ["a", "b", "c"], "inverse-frequency"), [20/36, 20/18, 20/6])

    def test_weight_order_and_unweighted_arm(self):
        counts = {"a": 3, "b": 1}
        self.assertEqual(subject.class_weights(counts, ["b", "a"], "inverse-frequency"), [2, 2/3])
        self.assertIsNone(subject.class_weights(counts, ["a", "b"], "none"))

    def test_invalid_class_counts_or_weighting_rejected(self):
        for counts in ({"a": 0, "b": 1}, {"a": -1, "b": 1}, {"a": 1.0, "b": 1}, {"a": 1}, {"a": 1, "b": 1, "c": 1}):
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    subject.class_weights(counts, ["a", "b"], "inverse-frequency")
        with self.assertRaisesRegex(ValueError, "Unknown weighting"):
            subject.class_weights({"a": 1, "b": 1}, ["a", "b"], "dev-frequency")

    def test_actual_training_call_uses_train_counts_only(self):
        calls = [node for node in ast.walk(function_node("train")) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "class_weights"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.dump(calls[0].args[0]), ast.dump(ast.parse("counts['train']", mode="eval").body))

    def test_hand_calculated_binary_confusion_metrics(self):
        report = subject.metric_summary([[8, 2], [1, 9]], 10., ["a", "b"])
        self.assertEqual(report["n_ROIs"], 20)
        self.assertEqual(report["confusion_orientation"], "rows=true, columns=predicted")
        self.assertAlmostEqual(report["accuracy"], 17/20)
        self.assertAlmostEqual(report["unweighted_NLL"], .5)
        self.assertAlmostEqual(report["balanced_accuracy"], .85)
        self.assertAlmostEqual(report["per_class"][0]["precision"], 8/9)
        self.assertAlmostEqual(report["per_class"][0]["recall"], 8/10)
        self.assertAlmostEqual(report["per_class"][0]["F1"], 16/19)
        self.assertAlmostEqual(report["per_class"][1]["precision"], 9/11)
        self.assertAlmostEqual(report["per_class"][1]["recall"], 9/10)
        self.assertAlmostEqual(report["per_class"][1]["F1"], 6/7)
        self.assertAlmostEqual(report["macro_F1"], (16/19+6/7)/2)

    def test_never_predicted_class_uses_explicit_zero_rule(self):
        report = subject.metric_summary([[2, 0, 0], [1, 0, 0], [0, 0, 3]], 6., ["a", "b", "c"])
        self.assertEqual(report["per_class"][1]["predicted_count"], 0)
        for name in ("precision", "recall", "F1"):
            self.assertEqual(report["per_class"][1][name], 0.)
        self.assertAlmostEqual(report["macro_F1"], .6)
        self.assertAlmostEqual(report["balanced_accuracy"], 2/3)

    def test_empty_or_malformed_confusion_rejected(self):
        with self.assertRaisesRegex(ValueError, "Empty evaluation"):
            subject.metric_summary([[0, 0], [0, 0]], 0., ["a", "b"])
        with self.assertRaisesRegex(ValueError, "Bad confusion shape"):
            subject.metric_summary([[1]], 1., ["a", "b"])

    def test_cpu_logits_evaluation_is_unweighted_and_keeps_record_ids(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized(), "A prior action initialized CUDA; CPU contract run must be isolated")
        probability = torch.tensor([[.8, .2], [.6, .4], [.4, .6], [.1, .9]], dtype=torch.float64, device="cpu")
        logits = probability.log()
        labels = torch.tensor([0, 0, 0, 1], dtype=torch.long, device="cpu")
        head = torch.nn.Identity().train()
        ids = ["fixture-0", "fixture-1", "fixture-2", "fixture-3"]
        report, predictions = subject.evaluate_head(head, logits, labels, ["a", "b"], ids)
        nlls = [-math.log(v) for v in (.8, .6, .4, .9)]
        expected = sum(nlls)/4
        class_weighted = ((2/3)*sum(nlls[:3]) + 2*nlls[3])/4
        self.assertAlmostEqual(report["unweighted_NLL"], expected, places=12)
        self.assertGreater(abs(expected-class_weighted), .01)
        self.assertEqual(report["confusion_matrix"], [[2, 1], [0, 1]])
        self.assertEqual([r["record_id"] for r in predictions], ids)
        self.assertEqual([r["predicted_id"] for r in predictions], [0, 0, 1, 1])
        self.assertFalse(head.training)
        self.assertFalse(torch.cuda.is_initialized())


class EpochBudgetContracts(unittest.TestCase):
    def test_fixed_full_data_update_budgets(self):
        self.assertEqual(subject.RECIPE["epochs"], 20)
        self.assertEqual(subject.RECIPE["batch_size"], 32)
        self.assertEqual(subject.expected_updates(1532), 960)
        self.assertEqual(subject.expected_updates(1276), 800)

    def test_update_count_keeps_partial_last_batch(self):
        for n, expected in ((1, 20), (31, 20), (32, 20), (33, 40), (64, 40), (65, 60)):
            with self.subTest(n=n):
                self.assertEqual(subject.expected_updates(n), expected)

    def test_production_epoch_range_and_resume_contract(self):
        expression = ast.fix_missing_locations(ast.Expression(copy.deepcopy(epoch_loop_node().iter)))
        code = compile(expression, str(TRAINER_PATH)+"::<epoch-range-only>", "eval")
        for completed in (0, 7, 19, 20):
            with self.subTest(completed=completed):
                actual = list(eval(code, {"completed": completed, "RECIPE": subject.RECIPE}))
                self.assertEqual(actual, list(range(completed+1, 21)))
        self.assertEqual(subject.RECIPE["selection"], "fixed_final_epoch_20")
        self.assertFalse(subject.RECIPE["early_stopping"])

    def test_actual_production_shuffle_and_tail_visit_every_roi_once(self):
        for n, batch_count, tail in ((1532, 48, 28), (1276, 40, 28), (33, 2, 1), (32, 1, 32)):
            with self.subTest(n=n):
                batches = production_epoch_indices(n, 20261106, 1)
                flat = [index for batch in batches for index in batch]
                self.assertEqual(len(batches), batch_count)
                self.assertEqual(len(batches[-1]), tail)
                self.assertTrue(all(1 <= len(batch) <= 32 for batch in batches))
                self.assertEqual(sorted(flat), list(range(n)))
                self.assertEqual(len(flat), len(set(flat)))
                self.assertEqual(batches, production_epoch_indices(n, 20261106, 1))
        self.assertNotEqual(production_epoch_indices(1532, 20261106, 1), production_epoch_indices(1532, 20261106, 2))
        self.assertEqual(subject.SEEDS, (20261106, 20261107, 20261108))
        self.assertEqual(subject.ARMS, ("none", "inverse-frequency"))



class FullSuiteCPUFixtureContract(unittest.TestCase):
    def test_synthetic_cpu_suite_budget_recompute_and_resume_integrity(self):
        import contextlib
        import io
        import tempfile
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        torch.set_num_threads(1)
        runtime_root = RESEARCH_ROOT / "tests" / "_runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        # The generated features and every output below are temporary UNIT TEST
        # fixtures, never actual dataset features or research experiment results.
        with tempfile.TemporaryDirectory(prefix="attribute_full_unit_", dir=runtime_root) as folder:
            root = Path(folder).resolve()
            self.assertTrue(root.is_relative_to(runtime_root.resolve()))
            cache = root / "synthetic_unit_cache"
            cache.mkdir()
            feature_file = cache / "features.pt"
            feature_file.write_bytes(b"SYNTHETIC UNIT FIXTURE ONLY - NOT A RESEARCH FEATURE CACHE")
            n_train, n_dev = 33, 4
            labels = torch.tensor([i % 2 for i in range(n_train)] + [0, 1, 0, 1], dtype=torch.long)
            features = torch.zeros((n_train+n_dev, 2048), dtype=torch.float32)
            features[torch.arange(len(labels)), labels] = 1.
            features[:, 2] = torch.arange(len(labels), dtype=torch.float32) / 100.
            rows = [{"split": "train" if i < n_train else "dev", "label_id": int(label),
                     "record_id": f"SYNTHETIC_UNIT_ROI_{i}"} for i, label in enumerate(labels.tolist())]
            payload = {"features": features, "labels": labels}
            complete = {"features_sha256": subject.sha(feature_file),
                        "backbone_tensor_sha256_after": "f" * 64,
                        "frozen_backbone_unchanged": True}
            identity = {"schema": "SYNTHETIC_UNIT_TEST_ONLY", "manifest_sha256": "c" * 64,
                        "script_sha256": subject.sha(TRAINER_PATH)}
            receipts = {"SYNTHETIC_UNIT_METADATA_NOT_A_SOURCE_PATH": "d" * 64}
            counts = {"train": {NAMES[0]: 17, NAMES[1]: 16}, "dev": {NAMES[0]: 2, NAMES[1]: 2}}
            document = fixture_document()
            args = SimpleNamespace(output=root / "synthetic_suite", resume=False, cache=cache)
            phases, integrity_verified = [], [False]
            real_write_json = subject.write_json

            def verified_inputs(value):
                self.assertEqual(value, receipts)
                integrity_verified[0] = True
                return dict(value)

            def observed_write(path, value):
                target = Path(path).resolve()
                self.assertTrue(target.is_relative_to(root))
                if target.name == "full_report.json":
                    phases.append(value["phase"])
                    if value["phase"] == "complete":
                        self.assertTrue(integrity_verified[0])
                real_write_json(target, value)

            def fixture_runtime(output, device):
                self.assertEqual(device, "cpu")
                target = Path(output).resolve()
                self.assertTrue(target.is_relative_to(root))
                target.mkdir(parents=True, exist_ok=True)

            with mock.patch.object(subject, "load_cache", return_value=(payload, complete)), \
                 mock.patch.object(subject, "verify_inputs", side_effect=verified_inputs), \
                 mock.patch.object(subject, "torch_setup", return_value=torch), \
                 mock.patch.object(subject, "configure_runtime", side_effect=fixture_runtime), \
                 mock.patch.object(subject, "write_json", side_effect=observed_write), \
                 contextlib.redirect_stdout(io.StringIO()):
                subject.train(args, document, rows, NAMES, receipts, counts, identity)
                summary_path = args.output / "suite_report.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self.assertEqual(summary["phase"], "complete")
                self.assertEqual(len(summary["results"]), 6)
                self.assertEqual(phases.count("pending_suite_integrity"), 6)
                self.assertEqual(phases.count("complete"), 6)
                self.assertFalse(summary["independent_test_claimed"])
                self.assertFalse(summary["test_used"])
                self.assertEqual(summary["evaluation_role"], "development_non_independent")
                stable_paths = [summary_path]
                checkpoint_hashes = {}
                for result in summary["results"]:
                    checkpoint_path, report_path = Path(result["checkpoint"]), Path(result["report"])
                    checkpoint_hashes[str(checkpoint_path)] = subject.sha(checkpoint_path)
                    stable_paths.extend([checkpoint_path, report_path, checkpoint_path.parent / "last.pt"])
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    self.assertEqual(report["actual_optimizer_updates"], 40)
                    self.assertEqual(report["actual_AdamW_parameter_steps"], [40, 40])
                    self.assertEqual(len(report["history"]), 20)
                    for epoch in report["history"]:
                        self.assertEqual(epoch["train_ROIs_used"], 33)
                        self.assertEqual([step["n_ROIs"] for step in epoch["steps"]], [32, 1])
                        self.assertTrue(all(step["gradient_finite"] for step in epoch["steps"]))
                    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    self.assertEqual(ck["completed_epoch"], 20)
                    self.assertEqual(ck["actual_optimizer_updates"], 40)
                    head = torch.nn.Linear(2048, 2)
                    head.load_state_dict(ck["head_state_dict"], strict=True)
                    metrics, _ = subject.evaluate_head(head, features[n_train:], labels[n_train:],
                        NAMES, [row["record_id"] for row in rows[n_train:]])
                    self.assertEqual(metrics, report["development_final"])
                before_resume = {str(path): subject.sha(path) for path in stable_paths}
                args.resume = True
                # Completed suite must verify and return, without any update or save.
                with mock.patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Resume retrained")), \
                     mock.patch.object(subject, "save_tensor", side_effect=AssertionError("Resume rewrote checkpoint")):
                    subject.train(args, document, rows, NAMES, receipts, counts, identity)
                self.assertEqual(before_resume, {str(path): subject.sha(path) for path in stable_paths})
                # Simulate only a missing final suite marker, within this disposable
                # test directory. Existing epoch-20 artifacts must be reused intact.
                self.assertTrue(summary_path.resolve().is_relative_to(root))
                summary_path.unlink()
                with mock.patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Epoch20 resume retrained")), \
                     mock.patch.object(subject, "save_tensor", side_effect=AssertionError("Epoch20 resume rewrote checkpoint")):
                    subject.train(args, document, rows, NAMES, receipts, counts, identity)
                self.assertEqual(checkpoint_hashes, {path: subject.sha(path) for path in checkpoint_hashes})
                self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8"))["phase"], "complete")
                # A field claiming the original hash must not conceal corrupted
                # head tensors when a partial suite reuses an epoch-20 artifact.
                summary_path.unlink()
                corrupt_path = Path(summary["results"][0]["checkpoint"])
                corrupt = torch.load(corrupt_path, map_location="cpu", weights_only=True)
                corrupt["head_state_dict"]["weight"][0, 0] += 1.
                torch.save(corrupt, corrupt_path)
                with mock.patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Corrupt resume retrained")), \
                     mock.patch.object(subject, "save_tensor", side_effect=AssertionError("Corrupt resume rewrote checkpoint")):
                    with self.assertRaisesRegex(ValueError, "Existing final checkpoint differs"):
                        subject.train(args, document, rows, NAMES, receipts, counts, identity)
        self.assertFalse(torch.cuda.is_initialized())


class FeatureCacheFinalizationCPUContracts(unittest.TestCase):
    """Actual I/O regression fixtures, never research data or feature evidence."""
    def setUp(self):
        import tempfile
        import torch
        self.torch = torch
        self.assertFalse(torch.cuda.is_initialized())
        runtime_root = RESEARCH_ROOT / "tests" / "_runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="attribute_finalize_unit_", dir=runtime_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(runtime_root.resolve()))
        self.case_count = 0

    def make_fixture(self):
        torch = self.torch
        self.case_count += 1
        output = self.root / f"SYNTHETIC_UNIT_CASE_{self.case_count}"
        output.mkdir()
        source = output / "UNIT_TEST_INPUT_NOT_RESEARCH.txt"
        source.write_text("Synthetic unit fixture only, never a research sample.", encoding="utf-8")
        rows = [{"record_index": i, "record_id": f"SYNTHETIC_UNIT_ROI_{i}",
                 "label_id": label, "split": "train" if i < 3 else "dev"}
                for i, label in enumerate([0, 1, 0, 0, 1])]
        identity = {"schema": subject.SCHEMA, "manifest_sha256": subject.sha(source),
                    "script_sha256": subject.sha(TRAINER_PATH), "class_names": NAMES.copy(),
                    "records": copy.deepcopy(rows)}
        features = torch.arange(5 * 2048, dtype=torch.float32).reshape(5, 2048) / 2048
        labels = torch.tensor([r["label_id"] for r in rows], dtype=torch.int64)
        receipts = {str(source): subject.sha(source)}
        runtime = {"device": "cpu", "torch": str(torch.__version__),
                   "feature_batch_size": 2, "purpose": "SYNTHETIC UNIT FIXTURE ONLY"}
        state = {"identity": identity, "runtime": runtime, "shards": [],
                 "input_sha256_before": receipts}
        for offset in (0, 2, 4):
            end = min(offset + 2, 5)
            name = f"shard_{offset:06d}.pt"
            piece = {"features": features[offset:end].clone(), "labels": labels[offset:end].clone(),
                     "record_ids": [r["record_id"] for r in rows[offset:end]],
                     "record_indices": list(range(offset, end))}
            subject.save_tensor(output / name, piece)
            state["shards"].append({"file": name, "sha256": subject.sha(output / name),
                                    "n": end-offset, "offset": offset})
        subject.write_json(output / "extract_state.json", state)
        kwargs = {"output": output, "state": state, "rows": rows, "identity": identity,
                  "receipts": receipts, "counts": {"train": {NAMES[0]: 2, NAMES[1]: 1},
                  "dev": {NAMES[0]: 1, NAMES[1]: 1}}, "document": fixture_document(),
                  "runtime": runtime, "before": "f" * 64, "after": "f" * 64, "no_grad": True}
        return SimpleNamespace(output=output, source=source, state=state, kwargs=kwargs,
                               features=features, labels=labels, identity=identity)

    def rewrite_piece(self, fixture, change, index=0):
        item = fixture.state["shards"][index]
        path = fixture.output / item["file"]
        piece = self.torch.load(path, map_location="cpu", weights_only=True)
        change(piece)
        subject.save_tensor(path, piece)
        item["sha256"] = subject.sha(path)

    def assert_refused(self, fixture, message):
        state_sha = subject.sha(fixture.output / "extract_state.json")
        with self.assertRaisesRegex(ValueError, message):
            subject.finalize_feature_cache(**fixture.kwargs)
        self.assertFalse((fixture.output / "complete.json").exists())
        self.assertFalse((fixture.output / "features.pt").exists())
        self.assertEqual(subject.sha(fixture.output / "extract_state.json"), state_sha)
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_real_shards_roundtrip_order_labels_features_and_complete_written_last(self):
        fixture = self.make_fixture()
        shard_hashes = {item["file"]: subject.sha(fixture.output / item["file"])
                        for item in fixture.state["shards"]}
        real_write = subject.write_json
        events = []
        def observe(path, value):
            self.assertEqual(Path(path), fixture.output / "complete.json")
            self.assertTrue((fixture.output / "features.pt").is_file())
            self.assertEqual(subject.sha(fixture.output / "features.pt"), value["features_sha256"])
            self.assertFalse(list(fixture.output.glob("*.tmp.*")))
            events.append("complete_after_atomic_features")
            return real_write(path, value)
        with mock.patch.object(subject, "write_json", side_effect=observe):
            complete = subject.finalize_feature_cache(**fixture.kwargs)
        payload, loaded_complete = subject.load_cache(fixture.output, fixture.identity, self.torch)
        self.assertEqual(events, ["complete_after_atomic_features"])
        self.assertEqual(complete, loaded_complete)
        self.assertTrue(self.torch.equal(payload["features"], fixture.features))
        self.assertTrue(self.torch.equal(payload["labels"], fixture.labels))
        self.assertEqual(payload["record_ids"], [r["record_id"] for r in fixture.kwargs["rows"]])
        self.assertEqual(complete["phase"], "complete")
        self.assertEqual(complete["label_semantics"], fixture.kwargs["document"]["label_semantics"])
        self.assertFalse(complete["test_used"])
        self.assertFalse(complete["independent_test_claimed"])
        self.assertEqual(shard_hashes, {name: subject.sha(fixture.output/name) for name in shard_hashes})
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_extract_calls_the_actual_finalize_function(self):
        calls = [node for node in ast.walk(function_node("extract")) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "finalize_feature_cache"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 11)

    def test_changed_shard_bytes_fail_before_completion(self):
        fixture = self.make_fixture()
        path = fixture.output / fixture.state["shards"][0]["file"]
        with path.open("ab") as stream:
            stream.write(b"INTENTIONAL UNIT TEST CORRUPTION")
        self.assert_refused(fixture, "shard bytes changed")

    def test_noncontiguous_missing_excess_and_escaping_shards_fail(self):
        cases = [
            ("offset", lambda f: f.state["shards"][1].update(offset=1), "noncontiguous"),
            ("zero", lambda f: f.state["shards"][0].update(n=0), "noncontiguous"),
            ("missing", lambda f: f.state["shards"].pop(), "missing records"),
            ("excess", lambda f: f.state["shards"][-1].update(n=2), "exceeds manifest"),
            ("escape", lambda f: f.state["shards"][0].update(file="../outside.pt"), "inside cache"),
            ("empty", lambda f: f.state["shards"].clear(), "empty cache"),
        ]
        for name, mutate, message in cases:
            with self.subTest(name=name):
                fixture = self.make_fixture()
                mutate(fixture)
                self.assert_refused(fixture, message)

    def test_record_indices_and_ids_are_checked_in_order(self):
        for field in ("record_indices", "record_ids"):
            with self.subTest(field=field):
                fixture = self.make_fixture()
                self.rewrite_piece(fixture, lambda p: p[field].reverse())
                self.assert_refused(fixture, "order/IDs differ")

    def test_label_values_dtype_and_shape_are_checked(self):
        torch = self.torch
        mutations = [lambda p: p.update(labels=p["labels"].flip(0)),
                     lambda p: p.update(labels=p["labels"].to(torch.int32)),
                     lambda p: p.update(labels=p["labels"].reshape(2, 1))]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                fixture = self.make_fixture()
                self.rewrite_piece(fixture, mutate)
                self.assert_refused(fixture, "shard labels differ")

    def test_feature_width_rows_dtype_and_nonfinite_are_checked(self):
        torch = self.torch
        mutations = [lambda p: p.update(features=p["features"][:, :2047]),
                     lambda p: p.update(features=p["features"][:1]),
                     lambda p: p.update(features=p["features"].to(torch.float64)),
                     lambda p: p["features"].__setitem__((0, 0), float("nan")),
                     lambda p: p["features"].__setitem__((0, 0), float("inf"))]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                fixture = self.make_fixture()
                self.rewrite_piece(fixture, mutate)
                self.assert_refused(fixture, "Invalid/nonfinite shard features")

    def test_identity_rows_and_labels_must_match(self):
        for field, value, message in (("record_id", "WRONG_UNIT_ID", "identity IDs differ"),
                                      ("label_id", 1, "identity labels differ")):
            with self.subTest(field=field):
                fixture = self.make_fixture()
                fixture.identity["records"][0][field] = value
                self.assert_refused(fixture, message)

    def test_changed_backbone_or_gradient_evidence_prevents_commit(self):
        for key, value in (("after", "e" * 64), ("no_grad", False)):
            with self.subTest(key=key):
                fixture = self.make_fixture()
                fixture.kwargs[key] = value
                self.assert_refused(fixture, "Backbone changed/received gradients")

    def test_changed_source_or_input_receipts_prevents_commit(self):
        fixture = self.make_fixture()
        fixture.identity["script_sha256"] = "0" * 64
        self.assert_refused(fixture, "Source changed")
        fixture = self.make_fixture()
        fixture.source.write_text("Changed synthetic unit source", encoding="utf-8")
        self.assert_refused(fixture, "Inputs changed")

    def test_feature_save_failure_never_writes_complete(self):
        fixture = self.make_fixture()
        with mock.patch.object(subject, "save_tensor", side_effect=OSError("unit fixture simulated disk failure")):
            with self.assertRaisesRegex(OSError, "simulated disk failure"):
                subject.finalize_feature_cache(**fixture.kwargs)
        self.assertFalse((fixture.output / "complete.json").exists())
        self.assertFalse((fixture.output / "features.pt").exists())
        self.assertTrue((fixture.output / "extract_state.json").is_file())

    def test_committed_cache_changed_bytes_are_rejected_by_actual_loader(self):
        fixture = self.make_fixture()
        subject.finalize_feature_cache(**fixture.kwargs)
        with (fixture.output / "features.pt").open("ab") as stream:
            stream.write(b"INTENTIONAL POSTCOMMIT UNIT CORRUPTION")
        with self.assertRaisesRegex(ValueError, "Feature cache bytes changed"):
            subject.load_cache(fixture.output, fixture.identity, self.torch)

    def test_actual_cpu_extract_from_four_fixture_images_and_local_official_backbone(self):
        import contextlib
        import io
        from PIL import Image
        torch = self.torch
        weights = Path("F:/研究生毕业设计/模块一定位补偿以及预测/apple_vision/local_amodal/cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth")
        self.assertTrue(weights.is_file(), "Expected authorized local official CPU fixture backbone")
        document = fixture_document()
        for index, row in enumerate(document["records"]):
            path = self.root / row["image_path"]
            Image.new("RGB", (32, 32), color=(index*50, 80+index*20, 150-index*30)).save(path)
            annotation = self.root / row["annotation_path"]
            annotation.write_text(json.dumps({"purpose": "SYNTHETIC UNIT FIXTURE ONLY", "index": index}), encoding="utf-8")
            row["image_sha256"] = subject.sha(path)
            row["annotation_sha256"] = subject.sha(annotation)
        manifest = self.root / "SYNTHETIC_UNIT_MANIFEST.json"
        subject.write_json(manifest, document)
        args = SimpleNamespace(manifest=manifest, weights=weights,
            allow_non_independent_development=True, output=self.root/"actual_cpu_extraction_unit",
            device="cpu", feature_batch_size=2, resume=False)
        torch.set_num_threads(4)
        with mock.patch.dict(os.environ, os.environ.copy()), \
             mock.patch.object(subject, "torch_setup", return_value=torch), \
             contextlib.redirect_stdout(io.StringIO()) as captured:
            values = subject.preflight(args)
            subject.extract(args, *values)
            identity = values[-1]
            payload, complete = subject.load_cache(args.output, identity, torch)
        self.assertIn("FEATURES 4/4", captured.getvalue())
        self.assertIn("FEATURE_CACHE_COMPLETE", captured.getvalue())
        self.assertEqual(tuple(payload["features"].shape), (4, 2048))
        self.assertEqual(payload["features"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(payload["features"]).all().item())
        self.assertEqual(payload["labels"].tolist(), [0, 1, 0, 1])
        self.assertEqual(payload["record_ids"], [r["record_id"] for r in identity["records"]])
        self.assertEqual(complete["runtime"]["device"], "cpu")
        self.assertEqual(complete["backbone_tensor_sha256_before"], complete["backbone_tensor_sha256_after"])
        self.assertTrue(complete["frozen_backbone_unchanged"])
        self.assertEqual(complete["input_sha256_before"], complete["input_sha256_after"])
        self.assertEqual(complete["label_semantics"], document["label_semantics"])
        self.assertFalse(complete["test_used"])
        self.assertFalse(complete["independent_test_claimed"])
        self.assertFalse(torch.cuda.is_initialized())

if __name__ == "__main__":
    unittest.main(verbosity=2)



