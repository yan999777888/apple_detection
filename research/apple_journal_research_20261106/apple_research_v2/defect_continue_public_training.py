"""Paired eight-epoch replay versus audited public-lesion transfer on YOLO26s."""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch

import defect9485_small as original
from class0_patch_refinement import ROOT, atomic_json, configure, read, sha, load_image
from defect_continue_public_transfer import (
    INTAKE,
    SSDA,
    PROBABILITY,
    draw_transfer,
    empirical_sizes,
    load_library,
    select_snapshot,
)
from joint_state_v1 import canonical_sha

OUT = ROOT / "experiments/defect_continue_public_v1"
PARENT = ROOT / "experiments/defect9485_small_v1/weights.pt"
PARENT_SHA = "21ca5879ab152ce754ed337e5f8abe3f1b611f99e3d86213be299816bf7f9de5"
LR = 0.00010
EPOCHS = 8
SEED = 948518
PRIMARY = [0.40, 0.34]


def parent():
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("Frozen 768px Small checkpoint differs")
    model = original.architecture()
    model.load_state_dict(
        torch.load(PARENT, map_location="cpu", weights_only=True), strict=True
    )
    return model


def protocol(mode, bank_receipt=None):
    train = original.samples("train")
    dev = original.samples("dev")
    stats = [(len(a), sum(len(r["instances"]) for r in a)) for a in (train, dev)]
    if (
        stats != [(596, 1276), (113, 230)]
        or original.SIZE != 768
        or original.PRIMARY != PRIMARY
    ):
        raise ValueError("SSDA scope, image size or fixed thresholds differ")
    return dict(
        mode=mode,
        parent_sha256=PARENT_SHA,
        ssda_manifest_sha256=sha(SSDA),
        code_sha256=sha(__file__),
        transfer_code_sha256=sha(
            Path(__file__).with_name("defect_continue_public_transfer.py")
        ),
        base_training_code_sha256=sha(original.__file__),
        train=stats[0],
        dev=stats[1],
        image_size=768,
        seed=SEED,
        epochs=EPOCHS,
        lr_peak=LR,
        learning_rate="two-epoch warmup, cosine decay ending at 5% of peak",
        optimizer="AdamW, weight decay 5e-4 on non-bias/non-norm weights; gradient norm limit 10",
        weights="eighth epoch EMA; no dev checkpoint or threshold selection",
        per_batch=original.BATCH,
        accumulation=original.ACCUM,
        expected_updates=400,
        thresholds=PRIMARY,
        inference="single-view FP32 with original class NMS 0.70",
        extra_rng="per (seed,epoch,original training row); separate from shuffle and original augment",
        transfer_probability=PROBABILITY if mode == "external" else 0.0,
        external=bank_receipt,
        prior_exposure="All 596 images trained the parent. Only the 113-image date development set is scored.",
    )


def transfer_rng(epoch, i):
    return np.random.default_rng(np.random.SeedSequence([SEED, epoch, int(i), 1337]))


def prepared_batch(images, labels, ids, bank, sizes, epoch):
    batch_im, batch_gt, inserted = [], [], []
    for i in ids:
        im, lab = images[i], labels[i]
        if bank is not None:
            im, lab, info = draw_transfer(im, lab, bank, sizes, transfer_rng(epoch, i))
            if info is not None:
                inserted.append(dict(host=int(i), **info))
        batch_im.append(im)
        batch_gt.append(lab)
    return batch_im, batch_gt, inserted


def preview_pair(im, result, box):
    side = 256
    l, t, r, b = box
    cx = (l + r) // 2
    cy = (t + b) // 2
    x = int(np.clip(cx - side // 2, 0, im.shape[1] - side))
    y = int(np.clip(cy - side // 2, 0, im.shape[0] - side))
    before = im[y : y + side, x : x + side]
    after = result[y : y + side, x : x + side]
    annotated = after.copy()
    cv2.rectangle(annotated, (l - x, t - y), (r - 1 - x, b - 1 - y), (68, 240, 240), 1)
    return np.hstack([before, after, annotated])


def audit(snapshot=None):
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_info = protocol("replay")
    model = parent()
    del model
    p = dict(
        parent_sha256=sha(PARENT),
        training=protocol_info["train"],
        development=protocol_info["dev"],
        cpu_parent_loaded_strict=True,
        device="cpu",
        external_snapshot_found=(INTAKE / "active.json").is_file(),
    )
    if not p["external_snapshot_found"]:
        p.update(
            external_ready=False,
            inserted=0,
            reason="No published, audited public polygon snapshot",
        )
        atomic_json(OUT / "preflight.json", p)
        return p
    bank, receipt = load_library(name=snapshot)
    audit_dir = OUT / "preflights" / receipt["snapshot"]
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows = original.samples("train")
    labels = [
        np.float32(
            [[v["class_id"], *v["xywh_normalized"]] for v in r["instances"]]
        ).reshape(-1, 5)
        for r in rows
    ]
    sizes = empirical_sizes(labels, original.SIZE)
    count, groups, examples, examples_per_class = Counter(), set(), [], Counter()
    sample_provenance = []
    preview_groups = set()
    for i, row in enumerate(rows):
        im = original.square(load_image(row["rgb_path"]))[0]
        gt = labels[i]
        result, new_gt, detail = draw_transfer(im, gt, bank, sizes, transfer_rng(0, i))
        if not np.array_equal(new_gt[: len(gt)], gt):
            raise ValueError("Original labels changed during transfer")
        if detail is None:
            continue
        c = detail["target_class"]
        count[c] += 1
        groups.add(detail["group"])
        if examples_per_class[c] < 4 and (c, detail["group"]) not in preview_groups:
            number = len(examples) + 1
            patch = preview_pair(im, result, detail["box"])
            label = f"{number} | {'scratch' if c==0 else 'pest damage'} | before / after / box"
            examples.append((label, patch))
            sample_provenance.append(
                dict(number=number, host=row["image_id"], **detail)
            )
            preview_groups.add((c, detail["group"]))
            examples_per_class[c] += 1
    if examples:
        panels = []
        for label, patch in examples:
            canvas = np.zeros((280, 768, 3), np.uint8)
            canvas[:256] = patch
            cv2.putText(
                canvas,
                label,
                (5, 274),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (230, 230, 230),
                1,
            )
            panels.append(canvas)
        sheet = np.vstack(panels)
        encoded = cv2.imencode(".jpg", sheet[:, :, ::-1])[1].tobytes()
        (OUT / "public_transfer_preview.jpg").write_bytes(encoded)
        (audit_dir / "public_transfer_preview.jpg").write_bytes(encoded)
    p.update(
        external_ready=bool(sum(count.values())),
        transfer_code_sha256=sha(
            Path(__file__).with_name("defect_continue_public_transfer.py")
        ),
        snapshot=receipt,
        preview_protocol="epoch 1, first four distinct donor groups per class, before/after/box in 256px native crops",
        preview_samples=sample_provenance,
        inserted=sum(count.values()),
        inserted_by_class=dict(count),
        groups_used=len(groups),
    )
    atomic_json(audit_dir / "preflight.json", p)
    atomic_json(OUT / "preflight.json", p)
    return p


def atomic_latest(path, payload):
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix="latest-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def exclusive_run(directory=OUT):
    """Disallow overlapping fits or benchmarks, and release the lock on process exit."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "gpu.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def publish(out, state, epochs, key):
    path = out / "weights.pt"
    if path.exists():
        # A power loss may leave the final epoch weights visible just before the receipt.
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.keys() != state.keys() or any(
            not torch.equal(saved[k], v) for k, v in state.items()
        ):
            raise ValueError("Published weights differ from the final EMA checkpoint")
    else:
        atomic_latest(path, state)
    atomic_json(
        out / "receipt.json",
        dict(
            weight_sha=sha(path),
            epochs=len(epochs),
            updates=epochs[-1]["updates"],
            protocol_sha256=key,
            parent_sha256=PARENT_SHA,
            inserted=sum(r["inserted"] for r in epochs),
        ),
    )


def load_resume(path, key, model, ema, optimizer, rng, criterion):
    if not path.is_file():
        return []
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p["protocol"] != key:
        raise ValueError("Resume requires the original protocol and source snapshot")
    model.load_state_dict(p["model"])
    ema.ema.load_state_dict(p["ema"])
    ema.updates = p["updates"]
    optimizer.load_state_dict(p["optimizer"])
    rng.bit_generator.state = p["rng"]
    torch.set_rng_state(p["torch_rng"])
    torch.cuda.set_rng_state(p["gpu_rng"])
    for _ in p["epochs"]:
        criterion.update()
    return p["epochs"]


def fit(mode, snapshot=None):
    from ultralytics.utils.loss import E2ELoss
    from ultralytics.utils.torch_utils import ModelEMA
    from joint_model_v1 import LossView

    bank, receipt = load_library(name=snapshot) if mode == "external" else (None, None)
    if mode == "external":
        check = read(OUT / "preflights" / receipt["snapshot"] / "preflight.json")
        if check["transfer_code_sha256"] != sha(
            Path(__file__).with_name("defect_continue_public_transfer.py")
        ):
            raise ValueError("Transfer implementation differs from CPU preflight")
        if (
            not check["external_ready"]
            or check["snapshot"]["manifest_sha256"] != receipt["manifest_sha256"]
        ):
            raise ValueError(
                "Run CPU preflight on this published snapshot before training"
            )
    p = protocol(mode, receipt)
    out = mode_folder(mode, receipt)
    out.mkdir(parents=True, exist_ok=True)
    key = canonical_sha(p)
    if (out / "receipt.json").exists():
        raise ValueError(f"Completed checkpoint already present at {out}")
    if (out / "weights.pt").exists() and not (out / "latest.pt").is_file():
        raise ValueError("Published weights exist without a recoverable final epoch")
    if (out / "protocol.json").exists() and canonical_sha(
        read(out / "protocol.json")
    ) != key:
        raise ValueError("Existing run has a different protocol")
    atomic_json(out / "protocol.json", p)
    images, labels = original.prepare()
    sizes = empirical_sizes(labels, original.SIZE)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    model = parent().cuda().train()
    ema = ModelEMA(model, decay=0.997, tau=200)
    crit = E2ELoss(LossView(model.model[-1], EPOCHS))
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
    opt = torch.optim.AdamW(groups, lr=LR)
    latest = out / "latest.pt"
    epochs = load_resume(latest, key, model, ema, opt, rng, crit)
    steps = epochs[-1]["updates"] if epochs else 0
    for e in range(len(epochs), EPOCHS):
        perm = rng.permutation(len(images))
        batches = np.array_split(perm, len(perm) // original.BATCH)
        losses, inserted = [], []
        for start in range(0, len(batches), original.ACCUM):
            group = batches[start : start + original.ACCUM]
            ep = e + start / len(batches)
            gain = (
                0.05
                + 0.95 * (1 + math.cos(math.pi * max(0, ep - 2) / (EPOCHS - 2))) / 2
            ) * min(1, (ep + 0.05) / 2)
            for params in opt.param_groups:
                params["lr"] = LR * gain
            opt.zero_grad(set_to_none=True)
            for ids in group:
                ims, gt, additions = prepared_batch(images, labels, ids, bank, sizes, e)
                inserted.extend(additions)
                x, y = original.augment(ims, gt, range(len(ids)), rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(x)
                pred = {
                    name: {
                        k: (
                            [t.float() for t in val]
                            if isinstance(val, list)
                            else val.float()
                        )
                        for k, val in values.items()
                    }
                    for name, values in pred.items()
                }
                loss = crit(pred, y)[0].sum() / len(ids)
                (loss / len(group)).backward()
                losses.append(loss.detach())
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
            mean_loss=float(torch.stack(losses).mean()),
            original_samples=len(perm),
            inserted=len(inserted),
            inserted_by_class=dict(Counter(v["target_class"] for v in inserted)),
        )
        if mode == "external" and not inserted:
            raise ValueError("No actual external positives consumed this epoch")
        atomic_json(out / f"inserted_epoch{e+1:02}.json", inserted)
        epochs.append(info)
        atomic_latest(
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
                epochs=epochs,
            ),
        )
        atomic_json(out / "training.json", epochs)
        print(info, flush=True)
    if steps != 400:
        raise ValueError(f"Unexpected update budget {steps}")
    publish(out, ema.ema.float().cpu().state_dict(), epochs, key)
    return out


def mode_folder(mode, receipt=None, snapshot=None):
    if mode == "replay":
        return OUT / "replay"
    if receipt is None:
        snap = select_snapshot(INTAKE, snapshot)
        receipt = read(snap / "receipt.json")
    return OUT / ("external_" + receipt["manifest_sha256"][:16])


@contextmanager
def evaluation_output(out):
    # The frozen historical evaluator uses this module's output directory for both
    # the weight input and reports. This CLI runs each evaluation in its own process.
    previous = original.OUT
    try:
        original.OUT = out
        yield
    finally:
        original.OUT = previous


def evaluate(mode, snapshot=None):
    out = mode_folder(mode, snapshot=snapshot)
    receipt = read(out / "receipt.json")
    if (
        receipt["parent_sha256"] != PARENT_SHA
        or receipt["updates"] != 400
        or (mode == "external" and not receipt["inserted"])
    ):
        raise ValueError("Incomplete training receipt")
    with evaluation_output(out):
        original.evaluate()
    result = read(out / "metrics.json")["fixed"]
    speed = read(out / "speed.json")
    if speed["n"] != 600:
        raise ValueError("Incomplete joint pipeline timing")
    return dict(fixed=result, joint_speed=speed, weight=str(out / "weights.pt"))


def compare(snapshot=None):
    parent_out = PARENT.parent
    replay, external = mode_folder("replay"), mode_folder("external", snapshot=snapshot)
    a, b = (read(p / "training.json") for p in (replay, external))
    if len(a) != EPOCHS or len(b) != EPOCHS:
        raise ValueError("Paired experiment incomplete")
    for x, y in zip(a, b):
        for k in (
            "epoch",
            "updates",
            "original_samples",
            "base_rng_after",
            "base_permutation",
        ):
            if x[k] != y[k]:
                raise ValueError(f"Unpaired training stream at epoch {x['epoch']}: {k}")
    if any(x["inserted"] != 0 for x in a) or any(y["inserted"] == 0 for y in b):
        raise ValueError("Invalid exposure controls")
    res = {}
    for tag, out in (
        ("original_parent", parent_out),
        ("replay", replay),
        ("external", external),
    ):
        scores = read(out / "metrics.json")["fixed"]
        speed = read(out / "speed.json")
        res[tag] = dict(
            scores=scores, speed=speed, weight_sha256=sha(out / "weights.pt")
        )
    atomic_json(
        OUT / "comparison.json",
        dict(
            fixed_score_thresholds=PRIMARY,
            original_aug_rng_matched=True,
            external_insertions=sum(y["inserted"] for y in b),
            modes=res,
        ),
    )
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["audit", "fit", "evaluate", "compare"])
    parser.add_argument("--mode", choices=["replay", "external"])
    parser.add_argument("--snapshot", help="Immutable intake snapshot directory name")
    args = parser.parse_args()
    if args.action in ("fit", "evaluate") and args.mode is None:
        parser.error("Choose --mode replay or external")
    cpu = args.action in ("audit", "compare")
    configure(OUT, "cpu" if cpu else "cuda:0")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if cpu:
        print(
            {"audit": audit, "compare": compare}[args.action](args.snapshot), flush=True
        )
    else:
        with exclusive_run():
            print(
                {"fit": fit, "evaluate": evaluate}[args.action](
                    args.mode, args.snapshot
                ),
                flush=True,
            )
