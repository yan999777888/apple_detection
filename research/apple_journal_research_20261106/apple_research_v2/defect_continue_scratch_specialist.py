"""Train one scratch specialist with a single shared, immutable insect backbone."""

import argparse
from pathlib import Path
import hashlib
import math
import cv2
import numpy as np
import torch
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import ModelEMA
from class0_patch_refinement import ROOT, atomic_json, configure, read, sha, load_image
from joint_model_v1 import LossView
from joint_state_v1 import canonical_sha
import defect_continue_public_training as replay
from defect_continue_scratch_net import (
    LockedInsect,
    ScratchHead,
    Deployed,
    WARM_SHA,
    REPLAY_SHA,
    objective,
    scratch_only,
    support,
    boxes,
)

OUT = ROOT / "experiments/defect_continue_scratch_specialist"
REF = replay.OUT / "replay"


def specialist():
    m = ScratchHead().cuda()
    m.head.stride = m.head.stride.cuda()
    return m


def check():
    torch.manual_seed(replay.SEED)
    locked = LockedInsect("cuda").eval()
    scratch = specialist().train()
    refs = replay.original.samples("train")
    with_scratches = [
        r for r in refs if any(g["class_id"] == 0 for g in r["instances"])
    ]
    mixed = [
        r for r in with_scratches if any(g["class_id"] == 1 for g in r["instances"])
    ]
    refs = [
        max(
            with_scratches,
            key=lambda r: sum(g["class_id"] == 0 for g in r["instances"]),
        ),
        max(mixed, key=lambda r: len(r["instances"])),
    ]
    imgs = [replay.original.square(load_image(r["rgb_path"]))[0] for r in refs]
    labels = [
        np.array(
            [[v["class_id"], *v["xywh_normalized"]] for v in r["instances"]],
            dtype=np.float32,
        ).reshape(-1, 5)
        for r in refs
    ]
    x, y = replay.original.augment(
        imgs, labels, range(len(refs)), np.random.default_rng(replay.SEED)
    )
    # Use the production single-image stride/BN path for the bit equality check.
    with torch.inference_mode():
        for frame in x:
            old = locked.warm(frame[None])
            fine, levels = locked.features(frame[None])
            if not torch.equal(old, locked.detections(levels)):
                raise ValueError("Insect output changed")
            if fine.shape[1:] != (128, 192, 192):
                raise ValueError("Unexpected P2")
    fine, levels = locked.features(x)
    crit = E2ELoss(LossView(scratch.head, replay.EPOCHS))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw = scratch(fine, levels)
    loss, parts, positives, pairs = objective(
        scratch, crit, raw, y, x.shape[-2:], len(x)
    )
    assert 0 < len(scratch_only(y)["cls"]) < len(y["cls"]) and positives > 0
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        scratch.parameters(), 10.0, error_if_nonfinite=True
    )
    assert not any(p.grad is not None for p in locked.parameters())
    assert not locked.training and all(not b.training for b in locked.modules())
    assert scratch.fine[-1].weight.grad.abs().sum() > 0
    assert scratch.head.one2one_cv3[0][-1].weight.grad.abs().sum() > 0
    logits = torch.tensor([[-1.0, 1.0, -0.5]], requires_grad=True)
    locations = torch.tensor([[[4.0, 4, 14, 14], [20, 20, 30, 30], [4.0, 4, 14, 14]]])
    ann = [torch.tensor([[4.0, 4, 14, 14, 0], [20.0, 20, 30, 30, 1]])]
    a, b, np_, ng = support(logits, locations, ann)
    (a + b).backward()
    assert (
        np_ == ng == 1
        and logits.grad[0, 0] == 0
        and logits.grad[0, 1] > 0
        and logits.grad[0, 2] < 0
    )
    data = dict(
        warm_sha=WARM_SHA,
        replay_sha=REPLAY_SHA,
        actual_single_frame_rows_bit_exact=600,
        train_images=[r["image_id"] for r in refs],
        actual_augmented_GT=len(y["cls"]),
        scratch_only_GT=len(scratch_only(y)["cls"]),
        frozen_BN_and_parameter_gradients_locked=True,
        p2_has_gradient=True,
        synthetic_single_positive_ranking_direction=True,
        loss=parts.tolist(),
        norm=float(norm),
        actual_protected=positives,
        actual_ranked=pairs,
    )
    atomic_json(OUT / "preflight.json", data)
    print(data, flush=True)


def protocol():
    p = replay.protocol("replay")
    p.update(
        mode="single backbone locked insects plus specialized scratch",
        locked_sha256=WARM_SHA,
        scratch_initial_sha256=REPLAY_SHA,
        net_sha256=sha(Path(__file__).with_name("defect_continue_scratch_net.py")),
        script_sha256=sha(__file__),
        class_thresholds=[0.40, 0.34],
        workpoint="last epoch 8 EMA, no development calibration",
        input="shared original 768 FP32 frozen features, scratch head training BF16, production FP32",
        additional_training="single-class native detection; per-GT recall floor .30 and same-image ranking .12 with known insect negatives",
        training_domains="596 original source training frames; no external images used",
    )
    return p


def fit():
    p = protocol()
    key = canonical_sha(p)
    if (OUT / "receipt.json").exists():
        raise ValueError("Finished training already exists")
    if not (OUT / "preflight.json").is_file():
        raise ValueError("Missing real gradient and invariance preflight")
    if (OUT / "protocol.json").exists() and canonical_sha(
        read(OUT / "protocol.json")
    ) != key:
        raise ValueError("Protocol changed")
    atomic_json(OUT / "protocol.json", p)
    rng = np.random.default_rng(replay.SEED)
    torch.manual_seed(replay.SEED)
    torch.backends.cudnn.benchmark = False
    images, labels = replay.original.prepare()
    locked = LockedInsect("cuda").eval()
    model = specialist().train()
    ema = ModelEMA(model, decay=0.997, tau=200)
    crit = E2ELoss(LossView(model.head, replay.EPOCHS))
    opt = torch.optim.AdamW(
        [
            dict(
                params=[
                    v for v in model.parameters() if v.requires_grad and v.ndim > 1
                ],
                weight_decay=0.0005,
            ),
            dict(
                params=[
                    v for v in model.parameters() if v.requires_grad and v.ndim <= 1
                ],
                weight_decay=0.0,
            ),
        ],
        lr=replay.LR,
    )
    latest = OUT / "latest.pt"
    history = replay.load_resume(latest, key, model, ema, opt, rng, crit)
    reference = read(REF / "training.json")
    steps = history[-1]["updates"] if history else 0
    for e in range(len(history), replay.EPOCHS):
        perm = rng.permutation(len(images))
        batches = np.array_split(perm, len(perm) // replay.original.BATCH)
        losses = []
        positive_count = pair_count = 0
        for start in range(0, len(batches), replay.original.ACCUM):
            group = batches[start : start + replay.original.ACCUM]
            progress = e + start / len(batches)
            gain = (
                0.05
                + 0.95
                * (1 + math.cos(math.pi * max(0, progress - 2) / (replay.EPOCHS - 2)))
                / 2
            ) * min(1, (progress + 0.05) / 2)
            for g in opt.param_groups:
                g["lr"] = replay.LR * gain
            opt.zero_grad(set_to_none=True)
            for ids in group:
                ims, gt, _ = replay.prepared_batch(images, labels, ids, None, None, e)
                x, target = replay.original.augment(ims, gt, range(len(ids)), rng)
                fine, pyramid = locked.features(x)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = model(fine, pyramid)
                loss, terms, np_, ng = objective(
                    model, crit, raw, target, x.shape[-2:], len(x)
                )
                (loss / len(group)).backward()
                losses.append(terms)
                positive_count += np_
                pair_count += ng
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 10.0, error_if_nonfinite=True
            )
            opt.step()
            ema.update(model)
            steps += 1
        crit.update()
        row = dict(
            epoch=e + 1,
            updates=steps,
            base_permutation=hashlib.sha256(perm.astype("<i8").tobytes()).hexdigest(),
            base_rng_after=canonical_sha(rng.bit_generator.state),
            original_samples=len(perm),
            inserted=0,
            native_positive_rank_loss=torch.stack(losses).mean(0).tolist(),
            actively_protected=positive_count,
            ranked_against_hard_negatives=pair_count,
        )
        for k in (
            "epoch",
            "updates",
            "base_permutation",
            "base_rng_after",
            "original_samples",
            "inserted",
        ):
            if row[k] != reference[e][k]:
                raise ValueError(f"Replay mismatch: {k}")
        history.append(row)
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
        print(row, flush=True)
    if steps != 400:
        raise ValueError("Unexpected optimizer update count")
    replay.atomic_latest(OUT / "weights.pt", ema.ema.float().cpu().state_dict())
    atomic_json(
        OUT / "receipt.json",
        dict(
            weight_sha=sha(OUT / "weights.pt"),
            locked_warm_sha=WARM_SHA,
            scratch_initial_sha=REPLAY_SHA,
            protocol_sha256=key,
            updates=steps,
            epochs=len(history),
            thresholds=[0.4, 0.34],
        ),
    )


def evaluate():
    receipt = read(OUT / "receipt.json")
    if receipt["protocol_sha256"] != canonical_sha(protocol()) or receipt[
        "weight_sha"
    ] != sha(OUT / "weights.pt"):
        raise ValueError("Integrity mismatch")

    def deploy():
        locked = LockedInsect("cuda").eval()
        s = specialist()
        s.load_state_dict(
            torch.load(OUT / "weights.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        return Deployed(locked, s.eval()).eval().requires_grad_(False)

    old = (replay.original.deploy_model, replay.original.boxes)
    try:
        replay.original.deploy_model = deploy
        replay.original.boxes = boxes
        with replay.evaluation_output(OUT):
            replay.original.evaluate()
    finally:
        replay.original.deploy_model, replay.original.boxes = old


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("check", "fit", "evaluate"))
    action = ap.parse_args().action
    configure(OUT, "cuda")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    with replay.exclusive_run(OUT):
        {"check": check, "fit": fit, "evaluate": evaluate}[action]()
