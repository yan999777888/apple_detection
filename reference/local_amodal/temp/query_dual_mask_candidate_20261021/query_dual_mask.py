"""UNEXECUTED candidate: shared Mask2Former queries, visible/amodal readouts.

Importing this file needs only the standard library. The factory would import
Torch only if a future, separately authorized runtime calls it. There is no
loader, download, optimizer, training CLI, implemented criterion or evaluator.
"""
from copy import deepcopy
from importlib.metadata import version

TRANSFORMERS_VERSION = "4.57.1"
MODEL_ID = "facebook/mask2former-swin-tiny-coco-instance"
MODEL_REVISION = "22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6"
EXPECTED_WEIGHTS_SHA256 = "3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80"
EXPECTED_AMODAL_MLP_PARAMETERS = 197376


def audit_outer_config(config):
    """Metadata only: do not inspect backbone_config.id2label as COCO labels."""
    if not isinstance(config, dict) or config.get("model_type") != "mask2former":
        raise ValueError("An outer Mask2Former config dictionary is required")
    labels = {int(k): str(v) for k, v in config.get("id2label", {}).items()}
    if set(labels) != set(range(80)):
        raise ValueError("Expected exactly the 80 outer COCO instance labels")
    if labels.get(47) != "apple" or config.get("label2id", {}).get("apple") != 47:
        raise ValueError("Official outer config must identify apple as index47")
    for key, expected in (("hidden_dim", 256), ("mask_feature_size", 256), ("num_queries", 100)):
        if config.get(key) != expected:
            raise ValueError(f"Unexpected audited model dimension: {key}")
    # HF creates num_labels+1 outputs and reserves the last row for no-object.
    return {"apple_source_row": 47, "source_no_object_row": len(labels),
            "source_classifier_outputs": len(labels)+1,
            "candidate_apple_label": 0, "candidate_no_object_label": 1,
            "hidden_dim": 256, "mask_feature_size": 256, "num_queries": 100,
            "new_amodal_MLP_parameters": 3*(256*256+256),
            "new_two_class_linear_parameters": 2*(256+1),
            "classification_probability_bitexact": False}


def dual_mask_hungarian_loss(*args, **kwargs):
    """Contract only; never substitute zero/fake losses for unfinished training."""
    raise NotImplementedError("Shared Hungarian assignment and real paired-mask criterion are not implemented")


def wrap_audited_pretrained(pretrained, *, source_identity):
    """Future factory for an already-loaded, independently verified HF object.

    This function does not open a file or fetch weights. source_identity is a
    caller assertion, not verification of the supplied object's tensor origin.
    A future loader must verify actual local bytes/strict loading separately.
    The original .model is shared by reference; a second backbone is not cloned.
    """
    if version("transformers") != TRANSFORMERS_VERSION:
        raise RuntimeError("This source audit covers Transformers4.57.1 only")
    if source_identity.get("model_id") != MODEL_ID or source_identity.get("revision") != MODEL_REVISION:
        raise ValueError("Unknown pretrained model identity/revision")
    if source_identity.get("local_weights_verified") is not True:
        raise ValueError("A future caller must separately verify local weight bytes")
    if source_identity.get("weights_sha256") != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Expected the separately verified official safetensors SHA256")
    if type(pretrained).__name__ != "Mask2FormerForUniversalSegmentation":
        raise TypeError("An official, already-loaded universal-segmentation object is required")
    specification = audit_outer_config(pretrained.config.to_dict())
    # Deliberately lazy: no Torch import has been executed during preparation.
    import torch
    from torch import nn

    source_classifier = pretrained.class_predictor
    if source_classifier.in_features != 256 or source_classifier.out_features != 81:
        raise ValueError("Official outer classifier must be Linear256->81")
    source_embedder = pretrained.model.transformer_module.decoder.mask_predictor.mask_embedder
    if sum(p.numel() for p in source_embedder.parameters()) != EXPECTED_AMODAL_MLP_PARAMETERS:
        raise ValueError("Unexpected official mask MLP structure")

    class QueryDualMaskCandidate(nn.Module):
        def __init__(self, shared, source_embedder, source_classifier, specification, source_identity):
            super().__init__()
            self.shared = shared
            self.amodal_mask_embedder = deepcopy(source_embedder)
            self.class_predictor = nn.Linear(256, 2, device=source_classifier.weight.device,
                                            dtype=source_classifier.weight.dtype)
            with torch.no_grad():
                for destination, source in ((0, specification["apple_source_row"]),
                                            (1, specification["source_no_object_row"])):
                    self.class_predictor.weight[destination].copy_(source_classifier.weight[source])
                    self.class_predictor.bias[destination].copy_(source_classifier.bias[source])
            if any(a.data_ptr() == v.data_ptr()
                   for a, v in zip(self.amodal_mask_embedder.parameters(), source_embedder.parameters())):
                raise RuntimeError("A/V mask embedding parameters must be independent")
            self.candidate_metadata = {
                "schema": "query_shared_visible_amodal_readout_candidate_v1",
                "source_identity_caller_assertion": deepcopy(source_identity),
                "source_tensor_identity_independently_verified_by_wrapper": False,
                "outer_class_mapping": deepcopy(specification),
                "classification_initialization": "apple47/null80 row copy; not81-class probability-preserving",
                "visible_path": "official shared.masks_queries_logits[-1] unchanged readout",
                "amodal_initialization": "independent copy of visible embedding MLP; not amodal-pretrained",
                "decoder_attention": "official visible-mask-generated attention; A readout does not steer attention",
                "criterion_implemented": False, "postprocess_implemented": False,
                "model_smoke_executed": False, "training_verified": False,
                "VRAM8GB_fit_verified": False, "quality_improvement_verified": False,
            }

        def forward(self, pixel_values, pixel_mask=None, *, targets=None):
            if targets is not None:
                raise NotImplementedError("No training-label adapter or dual-mask criterion has been implemented")
            # A future first runtime probe is batch1 with no external padding.
            # The audited base forward does not propagate pixel_mask throughout
            # the pixel/transformer modules; do not pretend padded-batch support.
            if pixel_values.ndim != 4 or pixel_values.shape[0] != 1 or pixel_values.shape[1] != 3:
                raise ValueError("Candidate contract is normalized RGB[B=1,3,H,W]")
            if pixel_mask is not None:
                if tuple(pixel_mask.shape) != (1, *pixel_values.shape[-2:]):
                    raise ValueError("pixel_mask shape mismatch")
                if not bool((pixel_mask == 1).all()):
                    raise NotImplementedError("Partial external padding requires an audited padding-aware adapter")
            outputs = self.shared(pixel_values=pixel_values, pixel_mask=pixel_mask,
                                  output_hidden_states=True, output_attentions=False, return_dict=True)
            intermediates = outputs.transformer_decoder_intermediate_states
            masks = outputs.masks_queries_logits
            pixels = outputs.pixel_decoder_last_hidden_state
            if not isinstance(intermediates, (tuple, list)) or not intermediates:
                raise RuntimeError("Missing public normalized intermediate decoder output")
            if not isinstance(masks, (tuple, list)) or len(masks) != len(intermediates):
                raise RuntimeError("Official visible-mask sequence must align with query stages")
            # Source implementation, not the misleading B/Q wording of some
            # docstrings: normalized intermediate is [Q,B,C], not [B,Q,C].
            normalized_qbc = intermediates[-1]
            if tuple(normalized_qbc.shape) != (100, 1, 256):
                raise RuntimeError("Unexpected normalized query feature shape")
            query_bqc = normalized_qbc.transpose(0, 1)
            if pixels.ndim != 4 or tuple(pixels.shape[:2]) != (1, 256):
                raise RuntimeError("Expected public pixel embeddings[B,256,h,w]")
            visible = masks[-1]
            expected = (1, 100, *pixels.shape[-2:])
            if tuple(visible.shape) != expected:
                raise RuntimeError("Official visible mask shape mismatch")
            amodal_embeddings = self.amodal_mask_embedder(query_bqc)
            amodal = torch.einsum("bqc,bchw->bqhw", amodal_embeddings, pixels)
            logits = self.class_predictor(query_bqc)
            return {"class_queries_logits": logits,
                    "visible_masks_queries_logits": visible,
                    "amodal_masks_queries_logits": amodal}

    return QueryDualMaskCandidate(pretrained.model, source_embedder, source_classifier,
                                  specification, source_identity)
