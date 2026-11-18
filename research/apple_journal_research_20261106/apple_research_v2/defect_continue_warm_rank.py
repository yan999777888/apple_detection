"""Parent-controlled quality/ranking ablation against the recorded 596-image replay."""

import argparse
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.tal import make_anchors
import defect_continue_public_training as replay
from defect_continue_rank_net import QualityDetector, auxiliary
from class0_patch_refinement import ROOT, read, sha, configure, atomic_json, load_image
from joint_model_v1 import LossView
from joint_state_v1 import canonical_sha

OUT = ROOT / "experiments/defect_continue_warm_rank"
REF = replay.OUT / "replay"


def initialized():
    if sha(replay.PARENT) != replay.PARENT_SHA:
        raise ValueError("Unexpected SSDA Small parent")
    m = QualityDetector()
    weights = torch.load(replay.PARENT, map_location="cpu", weights_only=True)
    m.base.load_state_dict(weights, strict=True)
    if not all(
        torch.count_nonzero(block[-1].weight) == 0
        and torch.count_nonzero(block[-1].bias) == 0
        for block in m.quality
    ):
        raise ValueError("Quality residual does not start at zero")
    return m


def unpack(y, hw, n):
    h, w = hw
    b = y["bboxes"].float() * y["bboxes"].new_tensor([w, h, w, h])
    corners = torch.cat(
        [b[:, :2] - 0.5 * b[:, 2:4], b[:, :2] + 0.5 * b[:, 2:4], y["cls"].float()],
        dim=1,
    )
    return [corners[y["batch_idx"].reshape(-1) == i] for i in range(n)]


def objective(model, crit, pred, q, y, hw):
    fp = {
        n: {
            k: ([t.float() for t in a] if isinstance(a, list) else a.float())
            for k, a in branch.items()
        }
        for n, branch in pred.items()
    }
    n = len(q)
    native = crit(fp, y)[0].sum() / n
    one = fp["one2one"]
    anchor, stride = make_anchors(one["feats"], model.head.stride, 0.5)
    box = (
        model.head.decode_bboxes(
            model.head.dfl(one["boxes"]), anchor.T.unsqueeze(0)
        ).transpose(1, 2)
        * stride
    )
    extra = auxiliary(box, one["scores"], q, unpack(y, hw, n))
    # Native box loss is identical to replay; the only added terms train quality and within-image ranking.
    total = native + 0.30 * extra["quality"] + 0.15 * extra["ranking"]
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite warm ranking objective")
    return total, (native, extra["quality"], extra["ranking"])


def cpu_check():
    from copy import deepcopy
    from defect_continue_rank_net import batch_targets

    cpu = torch.device("cpu")
    torch.manual_seed(replay.SEED)
    m = initialized().eval()
    base = replay.parent().eval()
    weights = base.state_dict()
    if weights.keys() != m.base.state_dict().keys() or not all(
        torch.equal(weights[k], v) for k, v in m.base.state_dict().items()
    ):
        raise ValueError("Parent tensors differ")
    rng = np.random.default_rng(replay.SEED)
    candidates = replay.original.samples("train")
    refs = [candidates[0], max(candidates[1:], key=lambda row: len(row["instances"]))]
    import cv2

    # Confirm the actual FP32 production resize and fused backbone, separately from
    # the smaller differentiable training check below. Use only fitting images.
    deployed = deepcopy(m)
    deployed.base.fuse(verbose=False)
    ref_deploy = deepcopy(base).fuse(verbose=False)
    deploy_rows = 0
    with torch.inference_mode():
        for row in refs:
            im, _ = replay.original.square(load_image(row["rgb_path"]))
            full = torch.from_numpy(im).permute(2, 0, 1)[None].float().div_(255)
            out = ref_deploy(full)[0]
            if not torch.equal(out, deployed(full)):
                raise AssertionError("Production-size fused parent predictions differ")
            deploy_rows += int(out.shape[1])
    del deployed, ref_deploy
    size = 512
    ims = np.stack(
        [cv2.resize(load_image(r["rgb_path"]), (size, size))[:, :, ::-1] for r in refs]
    ).copy()
    x = torch.from_numpy(ims).permute(0, 3, 1, 2).float().div_(255)
    with torch.inference_mode():
        orig = base(x)[0]
        augmented = m(x)
        if not torch.equal(orig, augmented):
            raise AssertionError("Warm output differs before training")
    annotations = []
    for r in refs:
        a = []
        for g in r["instances"]:
            cx, cy, w, h = g["xywh_normalized"]
            a.append(
                [
                    (cx - w / 2) * size,
                    (cy - h / 2) * size,
                    (cx + w / 2) * size,
                    (cy + h / 2) * size,
                    g["class_id"],
                ]
            )
        annotations.append(torch.tensor(a).reshape(-1, 5))
    y = batch_targets(annotations, (size, size), cpu)
    from_batch = unpack(y, (size, size), 2)
    if not all(
        torch.allclose(a, b, atol=2.0e-5) for a, b in zip(annotations, from_batch)
    ):
        raise AssertionError("Original augmented native coordinates/classes changed")
    native_loss = E2ELoss(LossView(base.model[-1], 8))
    crit = E2ELoss(LossView(m.head, 8))
    m.train()
    base.train()
    with torch.no_grad():
        ref_native = native_loss(base(x), y)[0].sum() / 2
    raw, q = m(x)
    total, terms = objective(m, crit, raw, q, y, (size, size))
    if not torch.equal(terms[0], ref_native):
        raise AssertionError("Existing native supervised loss differs")
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0, error_if_nonfinite=True)
    grads = [float(b[-1].weight.grad.abs().sum()) for b in m.quality]
    if not sum(grads) > 0 or m.base.model[0].conv.weight.grad.abs().sum() <= 0:
        raise ValueError("Missing real quality/backbone gradient")
    reference = read(REF / "training.json")
    perm = rng.permutation(596).astype("<i8")
    assert (
        hashlib.sha256(perm.tobytes()).hexdigest() == reference[0]["base_permutation"]
        and len(reference) == 8
    )
    report = dict(
        parent_sha256=replay.PARENT_SHA,
        base_state_keys=len(weights),
        cpu_images=2,
        image_ids=[row["image_id"] for row in refs],
        differentiable_input_size=size,
        original_output_rows=int(orig.shape[0] * orig.shape[1]),
        fused_production_size=replay.original.SIZE,
        fused_production_rows_verified=deploy_rows,
        fixed_output_bit_exact=True,
        native_loss_bit_exact=True,
        native_loss=float(ref_native),
        quality=float(terms[1].detach()),
        ranking=float(terms[2].detach()),
        real_gradient_quality=grads,
        gradient_norm=float(norm),
        first_reference_permutation_verified=True,
        replay_weight_sha256=read(REF / "receipt.json")["weight_sha"],
        fit=596,
        development=113,
        standalone_calibration=None,
    )
    atomic_json(OUT / "cpu_check.json", report)
    print(report, flush=True)


def fit():
    p = replay.protocol("replay")
    p.update(
        mode="warm_quality_rank",
        warm_code_sha256=sha(__file__),
        quality_code_sha256=sha(
            Path(__file__).with_name("defect_continue_rank_net.py")
        ),
        exact_replay_reference=sha(REF / "training.json"),
        loss_delta=dict(quality=0.3, within_image_ranking=0.15),
        primary="epoch8 EMA, original thresholds",
    )
    key = canonical_sha(p)
    if (OUT / "receipt.json").exists():
        raise ValueError("Finished warm ranking run exists")
    if (OUT / "protocol.json").exists() and canonical_sha(
        read(OUT / "protocol.json")
    ) != key:
        raise ValueError("Warm training protocol differs")
    atomic_json(OUT / "protocol.json", p)
    images, labels = replay.original.prepare()
    rng = np.random.default_rng(replay.SEED)
    torch.manual_seed(replay.SEED)
    torch.backends.cudnn.benchmark = False
    model = initialized().cuda().train()
    ema = ModelEMA(model, decay=0.997, tau=200)
    crit = E2ELoss(LossView(model.head, replay.EPOCHS))
    groups = [
        dict(
            params=[v for v in model.parameters() if v.requires_grad and v.ndim > 1],
            weight_decay=0.0005,
        ),
        dict(
            params=[v for v in model.parameters() if v.requires_grad and v.ndim <= 1],
            weight_decay=0.0,
        ),
    ]
    opt = torch.optim.AdamW(groups, lr=replay.LR)
    latest = OUT / "latest.pt"
    history = replay.load_resume(latest, key, model, ema, opt, rng, crit)
    steps = history[-1]["updates"] if history else 0
    reference = read(REF / "training.json")
    for e in range(len(history), replay.EPOCHS):
        perm = rng.permutation(len(images))
        batches = np.array_split(perm, len(perm) // replay.original.BATCH)
        losses = []
        for start in range(0, len(batches), replay.original.ACCUM):
            group = batches[start : start + replay.original.ACCUM]
            ep = e + start / len(batches)
            gain = (
                0.05
                + 0.95
                * (1 + math.cos(math.pi * max(0, ep - 2) / (replay.EPOCHS - 2)))
                / 2
            ) * min(1, (ep + 0.05) / 2)
            for g in opt.param_groups:
                g["lr"] = replay.LR * gain
            opt.zero_grad(set_to_none=True)
            for ids in group:
                ims, gt, inserted = replay.prepared_batch(
                    images, labels, ids, None, None, e
                )
                assert not inserted
                x, y = replay.original.augment(ims, gt, range(len(ids)), rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw, q = model(x)
                loss, terms = objective(model, crit, raw, q, y, x.shape[-2:])
                (loss / len(group)).backward()
                losses.append(torch.stack([t.detach() for t in terms]))
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 10.0, error_if_nonfinite=True
            )
            opt.step()
            ema.update(model)
            steps += 1
        crit.update()
        info = dict(
            epoch=e + 1,
            updates=steps,
            base_permutation=hashlib.sha256(perm.astype("<i8").tobytes()).hexdigest(),
            base_rng_after=canonical_sha(rng.bit_generator.state),
            original_samples=len(perm),
            inserted=0,
            mean_native_quality_ranking=torch.stack(losses).mean(0).tolist(),
        )
        for k in (
            "epoch",
            "updates",
            "base_permutation",
            "base_rng_after",
            "original_samples",
            "inserted",
        ):
            if info[k] != reference[e][k]:
                raise ValueError(f"Original replay stream diverged: {k}")
        history.append(info)
        replay.atomic_latest(
            latest,
            dict(
                protocol=key,
                model=model.state_dict(),
                ema=ema.ema.state_dict(),
                updates=steps,
                optimizer=opt.state_dict(),
                rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),
                gpu_rng=torch.cuda.get_rng_state(),
                epochs=history,
            ),
        )
        atomic_json(OUT / "training.json", history)
        print(info, flush=True)
    if steps != 400:
        raise ValueError("Unexpected warm rank budget")
    replay.publish(OUT, ema.ema.float().cpu().state_dict(), history, key)


def evaluate():
    receipt = read(OUT / "receipt.json")
    weight = OUT / "weights.pt"
    if (
        receipt["parent_sha256"] != replay.PARENT_SHA
        or sha(weight) != receipt["weight_sha"]
        or receipt["updates"] != 400
    ):
        raise ValueError("Warm ranking weight/protocol integrity failure")

    def deploy():
        m = initialized().eval()
        m.load_state_dict(
            torch.load(weight, map_location="cpu", weights_only=True), strict=True
        )
        m = m.cuda()
        m.base.fuse(verbose=False)
        return m.requires_grad_(False)

    orig = replay.original.deploy_model
    try:
        replay.original.deploy_model = deploy
        with replay.evaluation_output(OUT):
            replay.original.evaluate()
    finally:
        replay.original.deploy_model = orig


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["cpu-check", "fit", "evaluate"])
    mode = p.parse_args().action
    configure(OUT, "cpu" if mode == "cpu-check" else "cuda:0")
    torch.set_num_threads(1 if mode == "cpu-check" else 2)
    import cv2

    cv2.setNumThreads(1)
    if mode == "cpu-check":
        cpu_check()
    else:
        with replay.exclusive_run(OUT):
            {"fit": fit, "evaluate": evaluate}[mode]()
