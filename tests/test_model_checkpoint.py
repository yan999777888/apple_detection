"""CPU contract tests; real-parent integration is explicitly configured by env.

Set APPLE_SEG_PARENT_CHECKPOINT and APPLE_SEG_PARENT_SHA256 to test real bytes.
The actual run must set CUDA_VISIBLE_DEVICES=-1 before importing Torch.
"""
from copy import deepcopy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from apple_seg.model import (
    OUTPUT_KEYS, PARENT_RUN_ID, PARENT_SCHEMA, _strict_load_model_state,
    _validate_parent_payload, from_checkpoint,
)


def metadata_fixture():
    labels = {str(i): ("apple" if i == 47 else f"class{i}") for i in range(80)}
    return {"schema": PARENT_SCHEMA, "run_config": {"schema": PARENT_SCHEMA, "run_id": PARENT_RUN_ID},
            "completed_epoch": 6, "complete_train_epoch": True, "complete_full_val": True,
            "micro_batches_in_accumulation": 0, "model": {"fixture": torch.zeros(1)},
            "model_config": {"model_type": "mask2former", "num_queries": 100, "hidden_dim": 256,
                             "mask_feature_size": 256,
                             "backbone_config": {"model_type": "swin", "embed_dim": 96,
                                                 "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24]},
                             "id2label": labels, "label2id": {"apple": 47}}}


class CheckpointValidationTests(unittest.TestCase):
    def test_actual_parent_backbone_type_omission_is_preserved(self):
        original = metadata_fixture()
        original["model_config"]["backbone_config"].pop("model_type")
        audited = _validate_parent_payload(original)
        self.assertEqual(audited, original["model_config"])
        self.assertNotIn("model_type", audited["backbone_config"])

    def test_metadata_rejects_incomplete_or_wrong_parent(self):
        original = metadata_fixture()
        _validate_parent_payload(original)
        for key, value in [("complete_train_epoch", False), ("complete_full_val", False),
                           ("completed_epoch", 5), ("micro_batches_in_accumulation", 1)]:
            bad = deepcopy(original); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                _validate_parent_payload(bad)
        bad = deepcopy(original); bad["run_config"]["run_id"] = "other"
        with self.assertRaises(ValueError):
            _validate_parent_payload(bad)

    def test_config_rejects_dimensions_and_external_backbone(self):
        for key, value in [("num_queries", 99), ("hidden_dim", 128), ("mask_feature_size", 128),
                           ("use_pretrained_backbone", True), ("use_timm_backbone", True),
                           ("backbone", "remote/model")]:
            bad = metadata_fixture(); bad["model_config"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                _validate_parent_payload(bad)

    def test_hash_mismatch_prevents_unpickling(self):
        with patch("apple_seg.model.torch.load") as load:
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                from_checkpoint(Path(__file__), "0" * 64)
            load.assert_not_called()

    def test_digest_is_required(self):
        with self.assertRaises(ValueError):
            from_checkpoint("does-not-exist.pt", "")


@unittest.skipUnless(os.environ.get("APPLE_SEG_PARENT_CHECKPOINT") and os.environ.get("APPLE_SEG_PARENT_SHA256"),
                     "Configure the real parent checkpoint and expected SHA256")
class RealParentCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_initialized():
            raise RuntimeError("Real checkpoint tests require CUDA hidden and uninitialized")
        torch.set_num_threads(4)
        cls.path = Path(os.environ["APPLE_SEG_PARENT_CHECKPOINT"])
        cls.model, cls.provenance = from_checkpoint(cls.path, os.environ["APPLE_SEG_PARENT_SHA256"])
        cls.model.eval()

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_all_checkpoint_tensors_preserved_and_readouts_independent(self):
        payload = torch.load(self.path, map_location="cpu", weights_only=False, mmap=True)
        actual, original = self.model.state_dict(), payload["model"]
        self.assertEqual(self.provenance["model_config"], payload["model_config"])
        constructed = self.provenance["constructed_model_config"]
        self.assertEqual(constructed["backbone_config"]["model_type"], "swin")
        self.assertEqual((constructed["num_queries"], constructed["hidden_dim"], constructed["mask_feature_size"]),
                         (100, 256, 256))
        self.assertEqual(set(actual), set(original))
        self.assertEqual(self.provenance["strict_model_load"]["tensor_count"], len(original))
        for name, tensor in actual.items():
            self.assertEqual(tensor.dtype, original[name].dtype, name)
            self.assertTrue(torch.equal(tensor, original[name]), name)
        visible = self.model.shared.transformer_module.decoder.mask_predictor.mask_embedder
        for a, v in zip(self.model.amodal_mask_embedder.parameters(), visible.parameters(), strict=True):
            self.assertNotEqual(a.data_ptr(), v.data_ptr())
        self.assertFalse(self.provenance["optimizer_restored"])
        self.assertFalse(self.provenance["scaler_restored"])
        self.assertFalse(self.provenance["checkpoint_RNG_restored"])
        self.assertFalse(self.provenance["training_resume"])
        self.assertFalse(any(k.startswith("residual_box_head.") for k in actual))

    def test_one_backbone_forward_numerical_readouts_and_gradients_at64px(self):
        model = self.model; model.zero_grad(set_to_none=True)
        calls = []
        handle = model.shared.register_forward_hook(lambda _module, _args, result: calls.append(result))
        generator = torch.Generator(device="cpu").manual_seed(20261022)
        pixels = torch.randn((1, 3, 64, 64), generator=generator)
        try:
            outputs = model(pixels, torch.ones((1, 64, 64), dtype=torch.bool))
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(outputs), OUTPUT_KEYS)
        self.assertEqual(tuple(outputs["class_queries_logits"].shape), (1, 100, 2))
        self.assertEqual(tuple(outputs["visible_masks_queries_logits"].shape), (1, 100, 16, 16))
        self.assertEqual(tuple(outputs["amodal_masks_queries_logits"].shape), (1, 100, 16, 16))
        # Audit the original normalized final-QBC / official-visible readout path
        # from the single captured HF result, without another image forward.
        shared = calls[0]
        with torch.no_grad():
            query = shared.transformer_decoder_intermediate_states[-1].transpose(0, 1)
            expected_class = model.class_predictor(query)
            expected_amodal = torch.einsum("bqc,bchw->bqhw", model.amodal_mask_embedder(query),
                                          shared.pixel_decoder_last_hidden_state)
        self.assertTrue(torch.equal(outputs["class_queries_logits"], expected_class))
        self.assertTrue(torch.equal(outputs["visible_masks_queries_logits"], shared.masks_queries_logits[-1]))
        self.assertTrue(torch.equal(outputs["amodal_masks_queries_logits"], expected_amodal))
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in outputs.values()))
        # A synthetic contract objective exercises each readout, not accuracy or
        # the production TRAIN criterion. No optimizer step is performed.
        loss = sum(value.square().mean() for value in outputs.values())
        loss.backward()
        modules = {"A": model.amodal_mask_embedder,
                   "V": model.shared.transformer_module.decoder.mask_predictor.mask_embedder,
                   "class": model.class_predictor,
                   "shared": model.shared.pixel_level_module.encoder}
        for name, module in modules.items():
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads, name)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0, name)
        model.zero_grad(set_to_none=True)

    def test_strict_load_rejects_missing_tensor_before_mutation(self):
        state = dict(self.model.state_dict()); state.pop("class_predictor.bias")
        with self.assertRaisesRegex(RuntimeError, "Strict model keys differ"):
            _strict_load_model_state(self.model, state)

    def test_GT_and_partial_padding_rejected_without_backbone(self):
        pixels = torch.zeros(1, 3, 64, 64)
        with patch.object(self.model.shared, "forward") as forward:
            with self.assertRaises(ValueError):
                self.model(pixels, targets=[{}])
            with self.assertRaises(ValueError):
                self.model(pixels, torch.zeros(1, 64, 64))
            forward.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
