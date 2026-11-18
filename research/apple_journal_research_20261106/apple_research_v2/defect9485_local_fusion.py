"""Three fixed policies for routing disagreements between single-backbone lesion experts."""

import numpy as np
from class0_patch_refinement import ROOT, configure, atomic_json, read
from audit_defect_causes_20261112 import ious
from run_defect_zero_train_20261112 import metrics

OUT = ROOT / "experiments/defect9485_local_fusion"
ARMS = ("agreement", "deferred", "union")


def arrays(p):
    return np.asarray(p, np.float32).reshape(-1, 6)


def agrees(b, other):
    if len(b) == 0 or len(other) == 0:
        return np.zeros(len(b), bool)
    return (
        (ious(b[:, :4], other[:, :4]) >= 0.50) & (b[:, 5, None] == other[None, :, 5])
    ).any(1)


def add(old, new):
    new = new[~agrees(new, old)].copy()
    return np.concatenate((old, new))


def combine(a, b, c, high):
    # a: incumbent, b: native-scale surface classifier, c: learned instance selector.
    # Keep incumbent ordering; secondary boxes can supply previously missed lesions.
    kept_a = a[agrees(a, b) | agrees(a, c)]
    common_bc = b[agrees(b, c)]
    agreement = add(kept_a, common_bc)
    deferred = add(add(a, common_bc), high)
    union = add(add(a, b), high)
    return dict(agreement=agreement, deferred=deferred, union=union)


def run():
    folders = (
        "class0_crop_deploy_v1/fp32/dev",
        "defect9485_surface/joint",
        "defect9485_set/tradeoff",
        "defect9485_set/P94",
    )
    sources = [read(ROOT / f"experiments/{s}_predictions.json") for s in folders]
    rows = {k: [] for k in ARMS}
    for record in zip(*sources, strict=True):
        ref = record[0]
        assert all(
            (r["image_id"], r["GT"]) == (ref["image_id"], ref["GT"]) for r in record
        )
        preds = combine(*[arrays(r["pred"]) for r in record])
        for name, p in preds.items():
            rows[name].append(
                dict(image_id=ref["image_id"], GT=ref["GT"], pred=p.tolist())
            )
    for name, v in rows.items():
        atomic_json(OUT / f"{name}_predictions.json", v)
    met = {name: metrics(v) for name, v in rows.items()}
    atomic_json(OUT / "metrics.json", met)
    print({k: v["total"] for k, v in met.items()})


if __name__ == "__main__":
    configure(OUT, "cpu")
    run()
