"""CPU regression checks for masked donor geometry, frozen data and crash recovery."""

import argparse
import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from class0_patch_refinement import atomic_json, configure, read, sha
import defect_continue_public_training as experiment
import defect_continue_public_transfer as op


def rejects(fn, error=ValueError):
    try:
        fn()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def source_fixture(folder):
    snap = folder / "snapshots" / "checked"
    snap.mkdir(parents=True)
    full = np.full((80, 80, 3), [155, 109, 60], np.uint8)
    full[32:48, 33:47] = [57, 43, 22]
    Image.fromarray(full).save(snap / "original.png")
    crop = full[28:52, 28:52].copy()
    Image.fromarray(crop).save(snap / "crop.png")
    polygon = [[4, 4], [17, 4], [17, 19], [7, 15]]
    mask = Image.new("L", (24, 24))
    ImageDraw.Draw(mask).polygon([tuple(x) for x in polygon], fill=255)
    mask.save(snap / "mask.png")
    row = dict(
        source_id="source",
        archive_sha256="0" * 64,
        role="training_only_local_positive",
        background_supervision=False,
        license="CC BY 4.0",
        raw_split="train",
        mask_supervision=True,
        geometry="polygon",
        target_class=0,
        mask="mask.png",
        category_name="scratch",
        group_id="g",
        crop="crop.png",
        crop_sha256=sha(snap / "crop.png"),
        polygon_in_crop=[polygon],
        full_image="original.png",
        crop_xyxy=[28, 28, 52, 52],
        annotation=1,
    )
    receipt = dict(
        SSDA_manifest_sha256=sha(op.SSDA),
        source_specs=[
            dict(
                archive_sha256=row["archive_sha256"],
                license=row["license"],
                license_evidence_url="https://example.org/license",
                record_url="https://example.org/data",
                category_map={"scratch": 0},
            )
        ],
    )
    decisions = [
        dict(
            key="source",
            decision="export positive crops",
            group_id="g",
            sha256=sha(snap / "original.png"),
            raw_split="train",
        )
    ]

    def setrows(rows):
        atomic_json(snap / "positives.json", rows)
        atomic_json(
            snap / "receipt.json",
            dict(receipt, manifest_sha256=sha(snap / "positives.json")),
        )

    setrows([row])
    atomic_json(snap / "image_decisions.json", decisions)
    atomic_json(folder / "active.json", dict(snapshot="snapshots/checked"))
    return row, decisions, snap, setrows


def main():
    configure(experiment.OUT, "cpu")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    passed = []
    with tempfile.TemporaryDirectory(dir=experiment.OUT) as tmp:
        folder = Path(tmp)
        row, decisions, snap, setrows = source_fixture(folder)
        bank, receipt = op.load_library(folder)
        assert receipt["accepted_polygons_by_class"] == {0: 1}
        passed.append("train polygon hashed and geometry reconstructed")
        for change in (
            {"raw_split": "valid"},
            {"mask_supervision": False},
            {"mask": None},
            {"geometry": "bbox"},
        ):
            setrows([dict(row, **change)])
            rejects(lambda: op.load_library(folder))
        setrows([])
        rejects(lambda: op.load_library(folder))
        passed.append("empty list, held-out partition and missing native mask rejected")
        setrows([dict(row, target_class=1)])
        rejects(lambda: op.load_library(folder))
        passed.append("category map cannot be relabelled")
        setrows([row])
        atomic_json(
            snap / "image_decisions.json",
            decisions
            + [dict(key="other_capture_view", group_id="g", raw_split="test")],
        )
        assert op.load_library(folder)[1]["accepted_polygons_by_class"] == {0: 1}
        atomic_json(
            snap / "image_decisions.json", [dict(decisions[0], raw_split="test")]
        )
        rejects(lambda: op.load_library(folder))
        passed.append(
            "only official train image admitted; different camera view from same capture is not a second donor"
        )
        atomic_json(snap / "image_decisions.json", decisions)
        Image.new("L", (24, 24)).save(snap / "mask.png")
        rejects(lambda: op.load_library(folder))
        passed.append("altered mask fails polygon verification")
        donor = bank[0][0][0]
        host = np.full((160, 160, 3), [151, 106, 57], np.uint8)
        host[:24] = 0
        labels = np.array([[0, 0.48, 0.52, 0.20, 0.18]], dtype=np.float32)
        sizes = {0: np.array([[36.0, 30.0]]), 1: np.array([[18.0, 20.0]])}
        original_src, original_gt = host.copy(), labels.copy()

        def paste():
            return op.transfer(host, labels, donor, sizes, np.random.default_rng(17))

        out, gt, info = paste()
        l, t, r, b = info["box"]
        assert np.array_equal(labels, original_gt) and np.array_equal(
            host, original_src
        )
        assert np.array_equal(gt[:-1], labels) and int(gt[-1, 0]) == 0 and t >= 24
        assert np.any(out[t:b, l:r] != host[t:b, l:r])
        diff = np.any(out != host, axis=2)
        assert (
            not diff[:t].any()
            and not diff[b:].any()
            and not diff[:, :l].any()
            and not diff[:, r:].any()
        )
        assert not diff[68:99, 60:94].any()
        passed.append(
            "original RGB/GT untouched; no changes off the inserted box, on background or original GT"
        )
        assert np.array_equal(paste()[0], out)
        assert (
            op.transfer(
                np.zeros_like(host), labels, donor, sizes, np.random.default_rng(17)
            )
            is None
        )
        assert (
            op.transfer(
                host,
                np.float32([[0, 0.5, 0.5, 1.0, 1.0]]),
                donor,
                sizes,
                np.random.default_rng(17),
            )
            is None
        )
        passed.append(
            "reproducible; no eligible fruit surface or full GT overlap skips insertion"
        )
        # The exterior of a source polygon is never a supervised or pasted pixel,
        # even when interpolating on its boundary at a different crop size.
        contaminated = donor.rgb.copy()
        contaminated[~donor.mask] = [33, 251, 255]
        alt = op.Donor(
            donor.group, donor.key, donor.cls, contaminated, donor.mask, donor.skin
        )
        assert np.array_equal(
            out, op.transfer(host, labels, alt, sizes, np.random.default_rng(17))[0]
        )
        passed.append("changing all donor-background pixels cannot change pasted RGB")
        ctrl = np.random.default_rng(experiment.SEED)
        reference = np.random.default_rng(experiment.SEED)
        ids = ctrl.permutation(12)
        assert np.array_equal(reference.permutation(12), ids)
        experiment.prepared_batch([host] * 12, [labels] * 12, ids, bank, sizes, 2)
        assert ctrl.bit_generator.state == reference.bit_generator.state
        passed.append(
            "external placement consumes no shared original-augmentation RNG draws"
        )
        with experiment.exclusive_run(folder):

            def collide():
                with experiment.exclusive_run(folder):
                    pass

            rejects(collide, OSError)
        with experiment.exclusive_run(folder):
            pass
        passed.append("second job refused; released lock can be reacquired")
        final = folder / "final"
        final.mkdir()
        state = {"weight": torch.ones(2, 3)}
        epochs = [dict(updates=400, inserted=41)]
        experiment.publish(final, state, epochs, "test")
        receipt = read(final / "receipt.json")
        assert receipt["weight_sha"] == sha(final / "weights.pt")
        # Simulate an interruption after final weights were flushed.
        (final / "receipt.json").unlink()
        experiment.publish(final, state, epochs, "test")
        assert read(final / "receipt.json") == receipt
        rejects(
            lambda: experiment.publish(
                final, {"weight": torch.zeros(2, 3)}, epochs, "test"
            )
        )
        passed.append(
            "bare-state checkpoint read by original evaluator, crash recovery, mismatched EMA refused"
        )
    atomic_json(experiment.OUT / "cpu_checks.json", dict(passed=passed, device="cpu"))
    print(f"{len(passed)} checks passed")


def actual_training_batch(snapshot):
    """Exercise the real detection head and loss on one original/transplanted pair."""
    from ultralytics.utils.loss import E2ELoss
    from joint_model_v1 import LossView
    from class0_patch_refinement import load_image

    bank, receipt = op.load_library(name=snapshot)
    rows = experiment.original.samples("train")
    labels = [
        np.float32(
            [[v["class_id"], *v["xywh_normalized"]] for v in r["instances"]]
        ).reshape(-1, 5)
        for r in rows
    ]
    sizes = op.empirical_sizes(labels, experiment.original.SIZE)
    for idx, row in enumerate(rows):
        orig = experiment.original.square(load_image(row["rgb_path"]))[0]
        modified, gt, placed = op.draw_transfer(
            orig, labels[idx], bank, sizes, experiment.transfer_rng(0, idx)
        )
        if placed:
            break
    else:
        raise ValueError("Snapshot supplied no real training positives")
    before = experiment.sha(experiment.PARENT)
    torch.manual_seed(experiment.SEED)
    net = experiment.parent().train()
    criterion = E2ELoss(LossView(net.model[-1], experiment.EPOCHS))
    x = torch.from_numpy(np.stack([orig, modified])).permute(0, 3, 1, 2).float() / 255
    v = torch.from_numpy(np.concatenate([labels[idx], gt]))
    target = dict(
        cls=v[:, :1],
        bboxes=v[:, 1:],
        batch_idx=torch.repeat_interleave(
            torch.arange(2, dtype=torch.float32),
            torch.tensor([len(labels[idx]), len(gt)]),
        ),
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=experiment.LR)
    loss = criterion(net(x), target)[0].sum() / len(x)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        net.parameters(), 10.0, error_if_nonfinite=True
    )
    if not torch.isfinite(loss) or norm <= 0:
        raise ValueError("Nonfinite CPU loss or zero gradient")
    parameter = next(
        p for p in net.parameters() if p.grad is not None and p.grad.abs().max() > 0
    )
    prior = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(prior, parameter) and before == experiment.sha(
        experiment.PARENT
    )
    smoke = dict(
        device="cpu",
        dtype="float32",
        batch_images=2,
        image_size=int(x.shape[-1]),
        original_host=row["image_id"],
        native_labels=len(labels[idx]),
        new_labels=len(gt) - len(labels[idx]),
        snapshot=receipt["snapshot"],
        parent_sha=before,
        loss=float(loss.detach()),
        unmodified_gradient_norm=float(norm),
        one_update_in_memory=True,
    )
    out = experiment.OUT / "preflights" / receipt["snapshot"] / "cpu_training_step.json"
    atomic_json(out, smoke)
    print(smoke)


@torch.inference_mode()
def verify_initial_predictions(snapshot):
    """Rebuild each arm and compare its unmodified detections on eight training frames."""
    from class0_patch_refinement import load_image

    rows = experiment.original.samples("train")
    selection = [rows[i] for i in np.linspace(0, len(rows) - 1, 8).round().astype(int)]
    inputs = [
        torch.from_numpy(experiment.original.square(load_image(r["rgb_path"]))[0])
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255
        for r in selection
    ]
    bank, receipt = op.load_library(name=snapshot)
    protocol_a = experiment.protocol("replay")
    protocol_b = experiment.protocol("external", receipt)
    for field in (
        "parent_sha256",
        "seed",
        "expected_updates",
        "image_size",
        "thresholds",
    ):
        assert protocol_a[field] == protocol_b[field]
    outputs = []
    for mode in ("replay", "external"):
        torch.manual_seed(experiment.SEED)
        net = experiment.parent().eval().fuse(verbose=False)
        branch = []
        for x in inputs:
            raw = net(x)
            pred = raw[0][0] if isinstance(raw, tuple) else raw[0]
            assert pred.ndim == 2 and pred.shape[1] == 6 and torch.isfinite(pred).all()
            branch.append(pred.numpy().copy())
        outputs.append(branch)
        del net
    raw_digests = []
    for i, (first, second) in enumerate(zip(*outputs)):
        assert np.array_equal(first, second)
        raw_digests.append(
            dict(
                image=selection[i]["image_id"],
                raw_sha=hashlib.sha256(first.tobytes()).hexdigest(),
                candidates=len(first),
                fixed_kept=int(
                    (
                        first[:, 4]
                        >= np.array(experiment.PRIMARY)[first[:, 5].astype(int)]
                    ).sum()
                ),
            )
        )
    result = dict(
        device="cpu",
        precision="float32",
        seed=protocol_a["seed"],
        parent_sha=protocol_a["parent_sha256"],
        snapshot=receipt["snapshot"],
        identical_images=len(raw_digests),
        raw_boxes_and_scores_bit_identical=True,
        images=raw_digests,
    )
    atomic_json(
        experiment.OUT / "preflights" / receipt["snapshot"] / "paired_initial.json",
        result,
    )
    print(
        "identical initial detections on",
        len(raw_digests),
        "fixed training frames across both arms",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--training-step", metavar="SNAPSHOT")
    p.add_argument("--initial-predictions", metavar="SNAPSHOT")
    options = p.parse_args()
    main()
    if options.training_step:
        actual_training_batch(options.training_step)
    if options.initial_predictions:
        verify_initial_predictions(options.initial_predictions)
