"""Frozen CPU HDF author-image-class diagnostic. No fruit ROI or grading.
Requires a complete frozen identity audit. --check-only and --self-test never fit.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
 "schema": "hdf_peripheral_diagnostic_v1",
 "scope": "Four author image-level classes, real directory only; engineering development diagnostic, not fruit grading, physiological maturity, GT-ROI or independent external evaluation.",
 "source_caveat": "Author real directory includes online sources and contrast/rotation/crop augmentation; proxy groups do not establish independent fruit or capture identity.",
 "expected_source_records": 5000,
 "class_authority": "Use frozen manifest author_classes and author_class without relabeling.",
 "deduplication": "Lexicographically first path per (decoded_pixel_sha256, author_class); all original rows retain a mapping. Quarantine entire proxy group if any exact decoded-pixel identity has multiple author labels. Near duplicates are not collapsed, but stay in one proxy group. Also quarantine complete source-QC label-conflict families explicitly marked in frozen v2 identity audit, before any feature fit or development score.",
 "split": {"method": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True, "random_state": 20261106, "development_fold": 0, "retry_or_performance_selection": False},
 "image": {"orientation": "PIL ImageOps.exif_transpose", "alpha": "require opaque; never invent a background", "resize": [256, 256], "interpolation": "PIL bilinear", "preserve_aspect_ratio": False, "crop": False},
 "variants": [{"name": "whole", "border_px": None, "role": "whole-image comparator"}, {"name": "peripheral_8", "border_px": 8, "role": "primary peripheral-only diagnostic"}, {"name": "peripheral_16", "border_px": 16, "role": "prespecified sensitivity, not model selection"}],
 "features": {"dimension": 71, "rgb": "mean,std,q10,q50,q90 per channel", "hsv": "12-bin normalized hue histogram; mean,std,q10,q50,q90 for saturation/value", "gray": "16-bin normalized histogram", "texture": "Sobel(ksize3) magnitude 5 statistics + magnitude-weighted 8-bin orientation histogram; absolute Laplacian(ksize3) 5 statistics", "region": "Direct selected-pixel statistics, no center filling. Texture uses 3x3-eroded region, excluding inner and outer boundaries so every kernel sample lies inside the selected region.", "histogram_ranges": {"hue": [0, 360], "gray": [0, 1], "orientation": [-3.141592653589793, 3.141592653589793]}, "normalization": "float64 statistics of float32 RGB in [0,1]; sum-normalized histograms, all-zero orientation histogram if all gradients zero"},
 "classifier": {"standardization": "StandardScaler fitted only on training representatives", "estimator": "sklearn.linear_model.LogisticRegression", "C": 1.0, "l1_ratio": 0.0, "solver": "lbfgs", "class_weight": "balanced", "max_iter": 2000, "tol": 0.000001, "fits_per_variant": 1, "random_state": None, "threads": 1, "reason_single_fit": "LBFGS is deterministic for fixed data; no invented multi-seed mean or uncertainty.", "hyperparameter_search": False, "convergence_warning": "fail; no silent acceptance or development tuning"},
 "evaluation": {"primary": "Exact-pixel-deduplicated representative-image metrics on fixed development fold", "metrics": ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "multiclass_log_loss", "per_class_precision_recall_f1_support", "confusion_matrix", "mean_proxy_group_accuracy"], "confidence_interval": None, "confusion_axes": "rows=true author class, columns=predicted author class", "train_metrics": "separate fitting diagnostics", "comparison": "Peripheral-minus-whole differences are descriptive; never choose width, split or classifier from development performance.", "interpretation_limit": "Periphery may contain fruit; peripheral class association does not prove background causality. No segmentation, fruit-level, hidden-quality or population-generalization claim."},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def f_path(value, within_root=False):
    p = Path(value).resolve()
    if p.drive.upper() != "F:" or (within_root and not p.is_relative_to(ROOT)):
        raise ValueError(f"F-only research path required: {p}")
    return p


def write_json(path, value):
    with f_path(path, True).open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def configure_environment():
    cache = ROOT / "runtime" / "hdf_cpu_diagnostic"
    for key, name in {"TEMP": "tmp", "TMP": "tmp", "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "MPLCONFIGDIR": "matplotlib", "JOBLIB_TEMP_FOLDER": "joblib", "PYTHONPYCACHEPREFIX": "pycache", "HF_HOME": "hf", "TORCH_HOME": "torch", "CUDA_CACHE_PATH": "cuda", "PIP_CACHE_DIR": "pip"}.items():
        os.environ[key] = str(cache / name)
    os.environ.update({"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false", "CUDA_VISIBLE_DEVICES": "-1"})
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = "1"
    sys.dont_write_bytecode = True


def load_manifest(path, expected):
    if sha256(path) != expected.lower():
        raise ValueError("Grouping manifest differs from explicitly frozen expected SHA")
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    if doc.get("status") != "complete" or not doc.get("schema") or not doc.get("source_authority"):
        raise ValueError("A complete identity audit with schema/source authority is mandatory")
    classes, rows = doc.get("author_classes"), doc.get("records")
    if not isinstance(classes, list) or len(classes) != 4 or not all(isinstance(c, str) for c in classes) or len(set(classes)) != 4:
        raise ValueError("author_classes must be four unique strings")
    if not isinstance(rows, list) or len(rows) != PROTOCOL["expected_source_records"]:
        raise ValueError("Expected all 5000 frozen author records")
    by_pixel, by_byte, source = defaultdict(list), defaultdict(list), []
    real_root = ROOT / "apple_research_v2/sources/additional_attribute_sources/hdf_real_only/raw"
    keys = ("image_path", "sha256", "decoded_pixel_sha256", "author_class", "recommended_proxy_group_id")
    for i, row in enumerate(rows):
        if any(not row.get(k) for k in keys):
            raise ValueError(f"Missing identity at record {i}")
        p = f_path(row["image_path"])
        if not p.is_relative_to(real_root.resolve()) or row["author_class"] not in classes:
            raise ValueError("Image outside fixed real-only source or undeclared author label")
        for key in ("sha256", "decoded_pixel_sha256"):
            if len(row[key]) != 64 or any(c not in "0123456789abcdef" for c in row[key]):
                raise ValueError(f"Invalid {key}")
        if not isinstance(row["recommended_proxy_group_id"], str):
            raise ValueError("Proxy group ID must be a string")
        rec = {k: row[k] for k in keys}
        for key in ("label_conflict_family", "training_eligible", "quarantined"):
            if not isinstance(row.get(key), bool):
                raise ValueError("Frozen v2 source-QC boolean flags are mandatory")
        if row["training_eligible"] == row["quarantined"] or (row["label_conflict_family"] and not row["quarantined"]):
            raise ValueError("Inconsistent source-QC quarantine flags")
        rec.update(image_path=str(p), source_record_index=i, source_training_eligible=row["training_eligible"], source_audit_quarantined=row["quarantined"], source_label_conflict_family=row["label_conflict_family"], source_quarantine_reason=row.get("quarantine_reason"))
        source.append(rec)
        by_pixel[rec["decoded_pixel_sha256"]].append(rec)
        by_byte[rec["sha256"]].append(rec)
    for kind, grouped in (("byte", by_byte), ("decoded pixel", by_pixel)):
        if any(len({r["recommended_proxy_group_id"] for r in rr}) != 1 for rr in grouped.values()):
            raise ValueError(f"Audit splits exact {kind} duplicates across proxy groups")
    conflicts = {r["recommended_proxy_group_id"] for rr in by_pixel.values() if len({r["author_class"] for r in rr}) > 1 for r in rr}
    source_quarantine = {r["recommended_proxy_group_id"] for r in source if r["source_audit_quarantined"]}
    if any(not r["source_audit_quarantined"] for r in source if r["recommended_proxy_group_id"] in source_quarantine):
        raise ValueError("Source audit did not quarantine a complete proxy group")
    conflicts |= source_quarantine
    buckets = defaultdict(list)
    for row in source:
        buckets[(row["decoded_pixel_sha256"], row["author_class"])].append(row)
    representatives, mappings = [], []
    for rr in sorted(buckets.values(), key=lambda records: min(r["image_path"] for r in records)):
        rr.sort(key=lambda r: (r["image_path"], r["source_record_index"]))
        representative = dict(rr[0])
        rid = len(representatives)
        representative.update(representative_id=rid, occurrence_count=len(rr), quarantined=representative["recommended_proxy_group_id"] in conflicts)
        representatives.append(representative)
        mappings.extend({**r, "representative_id": rid, "quarantined": representative["quarantined"]} for r in rr)
    return doc, source, representatives, mappings, conflicts


def verify_sources(source):
    seen = {}
    for i, row in enumerate(source):
        p, expected = row["image_path"], row["sha256"]
        if p in seen and seen[p] != expected:
            raise ValueError("Same path has conflicting expected bytes")
        if p not in seen:
            if sha256(p) != expected:
                raise ValueError(f"Source bytes changed: {p}")
            seen[p] = expected
        if (i + 1) % 1000 == 0:
            print(json.dumps({"phase": "verify_source", "records": i + 1}), flush=True)
    return len(seen)


def freeze_split(representatives, classes):
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    rows = [r for r in representatives if not r["quarantined"]]
    y = np.asarray([classes.index(r["author_class"]) for r in rows], dtype=np.int64)
    groups = np.asarray([r["recommended_proxy_group_id"] for r in rows])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261106)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tr, dv = next(splitter.split(np.zeros((len(rows), 1)), y, groups))
    if set(y[tr]) != set(range(4)) or set(y[dv]) != set(range(4)):
        raise ValueError("Fixed first fold lacks a class; no automatic reseeding or split selection")
    if set(groups[tr]) & set(groups[dv]):
        raise AssertionError("Proxy-group leakage")
    for r in representatives:
        r["split"] = "quarantine" if r["quarantined"] else None
    for ids, name in ((tr, "train"), (dv, "development")):
        for i in ids:
            rows[int(i)]["split"] = name
    summary = {}
    for name in ("train", "development", "quarantine"):
        part = [r for r in representatives if r["split"] == name]
        summary[name] = {"representatives": len(part), "source_occurrences": sum(r["occurrence_count"] for r in part), "proxy_groups": len({r["recommended_proxy_group_id"] for r in part}), "class_support": {c: sum(r["author_class"] == c for r in part) for c in classes}}
    return rows, y, groups, tr, dv, summary


def load_rgb(path):
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as img:
        oriented = ImageOps.exif_transpose(img)
        if oriented.convert("RGBA").getchannel("A").getextrema() != (255, 255):
            raise ValueError(f"Non-opaque source requires unapproved compositing: {path}")
        rgb = oriented.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def regions():
    import cv2
    import numpy as np
    output = []
    for spec in PROTOCOL["variants"]:
        mask = np.ones((256, 256), dtype=np.uint8)
        w = spec["border_px"]
        if w is not None:
            mask[w:-w, w:-w] = 0
        texture = cv2.erode(mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        output.append((spec["name"], mask.astype(bool), texture.astype(bool)))
    return output


def feature_names():
    stat = ("mean", "std", "q10", "q50", "q90")
    names = [f"{c}_{s}" for c in ("red", "green", "blue") for s in stat]
    names += [f"hue_hist_{i:02d}" for i in range(12)]
    names += [f"{c}_{s}" for c in ("saturation", "value") for s in stat]
    names += [f"gray_hist_{i:02d}" for i in range(16)]
    names += [f"sobel_magnitude_{s}" for s in stat]
    names += [f"sobel_orientation_hist_{i:02d}" for i in range(8)]
    names += [f"absolute_laplacian_{s}" for s in stat]
    assert len(names) == 71
    return names


def describe(rgb):
    import cv2
    import numpy as np
    if rgb.shape != (256, 256, 3) or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Invalid real-image RGB preprocessing")
    hsv, gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dx, dy = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = np.hypot(dx, dy), np.arctan2(dy, dx)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    def stats(v):
        v = v.astype(np.float64, copy=False)
        return [float(v.mean()), float(v.std()), *np.quantile(v, [0.1, 0.5, 0.9]).tolist()]
    def hist(v, bins, span, weights=None):
        h = np.histogram(v, bins=bins, range=span, weights=weights)[0].astype(np.float64)
        return (h / h.sum() if h.sum() > 0 else h).tolist()
    result = {}
    for name, mask, texture in regions():
        values = sum([stats(rgb[:, :, k][mask]) for k in range(3)], [])
        values += hist(hsv[:, :, 0][mask], 12, (0, 360))
        values += sum([stats(hsv[:, :, k][mask]) for k in (1, 2)], [])
        values += hist(gray[mask], 16, (0, 1))
        values += stats(mag[texture])
        values += hist(angle[texture], 8, (-np.pi, np.pi), mag[texture].astype(np.float64))
        values += stats(lap[texture])
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (71,) or not np.isfinite(vector).all():
            raise ValueError("Non-finite or incorrect feature vector")
        result[name] = vector
    return result


def runtime_record():
    import cv2
    import numpy as np
    import scipy
    import sklearn
    import PIL
    import sklearn.linear_model._logistic as logistic
    import sklearn.model_selection._split as split
    import sklearn.preprocessing._data as preprocess
    import sklearn.metrics._classification as metrics
    from threadpoolctl import threadpool_info
    return {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__, "threadpools": threadpool_info(), "local_source_sha256": {m.__name__: {"path": m.__file__, "sha256": sha256(m.__file__)} for m in (logistic, split, preprocess, metrics)}, "gpu_imported": gpu_imported()}


def gpu_imported():
    return any(k == "torch" or k.startswith("ultralytics") or k.startswith("onnxruntime") for k in sys.modules)


def evaluate(y, probabilities, groups, classes):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
    prediction = probabilities.argmax(axis=1)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-10):
        raise ValueError("Invalid probabilities")
    p, r, f, n = precision_recall_fscore_support(y, prediction, labels=np.arange(4), zero_division=0)
    ga = [float((prediction[groups == g] == y[groups == g]).mean()) for g in sorted(set(groups))]
    return {"n_representative_images": len(y), "n_proxy_groups": len(ga), "accuracy": float(accuracy_score(y, prediction)), "balanced_accuracy": float(balanced_accuracy_score(y, prediction)), "macro_f1": float(f1_score(y, prediction, labels=np.arange(4), average="macro", zero_division=0)), "weighted_f1": float(f1_score(y, prediction, labels=np.arange(4), average="weighted", zero_division=0)), "multiclass_log_loss": float(log_loss(y, probabilities, labels=np.arange(4))), "mean_proxy_group_accuracy": float(np.mean(ga)), "per_class": [{"author_class": c, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(n[i])} for i, c in enumerate(classes)], "confusion_matrix": confusion_matrix(y, prediction, labels=np.arange(4)).tolist(), "class_order": classes, "zero_division": 0}


def self_test(rows, groups, tr, dv):
    import numpy as np
    assert not set(groups[tr]) & set(groups[dv])
    masks = regions()
    assert [int(m.sum()) for _, m, _ in masks] == [65536, 7936, 15360]
    for name, mask, texture in masks:
        assert np.all(~texture | mask)
        assert not texture[0].any() and not texture[-1].any() and not texture[:, 0].any() and not texture[:, -1].any()
        yy, xx = np.where(texture)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                assert mask[yy + dy, xx + dx].all()
    assert not masks[1][1][8:-8, 8:-8].any() and not masks[2][1][16:-16, 16:-16].any()
    samples = []
    for label in sorted({r["author_class"] for r in rows}):
        row = next(r for r in rows if r["author_class"] == label)
        first, second = describe(load_rgb(row["image_path"])), describe(load_rgb(row["image_path"]))
        for name in first:
            assert np.array_equal(first[name], second[name])
            for sl in (slice(15, 27), slice(37, 53)):
                assert abs(first[name][sl].sum() - 1) < 1e-12
        samples.append({"path": row["image_path"], "sha256": row["sha256"], "author_class": label})
    assert not gpu_imported()
    return {"passed": True, "tests": ["fixed group-disjoint split", "whole/ring pixel counts", "all 3x3 texture samples inside region", "center excluded", "real-image finite 71D descriptors", "repeat real-image features deterministic", "histograms normalized", "no GPU/segmentation module import"], "real_source_samples": samples}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grouping-manifest", type=Path, required=True)
    ap.add_argument("--expected-manifest-sha256", required=True)
    ap.add_argument("--protocol", type=Path, default=Path(__file__).with_name("hdf_peripheral_protocol_v1.json"))
    ap.add_argument("--output", type=Path)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    configure_environment()
    import numpy as np
    import cv2
    from threadpoolctl import threadpool_limits
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cv2.setNumThreads(1)
    manifest, protocol_path = f_path(args.grouping_manifest), f_path(args.protocol, True)
    if json.loads(protocol_path.read_text(encoding="utf-8-sig")) != PROTOCOL:
        raise ValueError("Protocol does not match frozen runner")
    doc, source, representatives, mappings, conflicts = load_manifest(manifest, args.expected_manifest_sha256)
    unique_paths = verify_sources(source)
    rows, y, groups, tr, dv, summary = freeze_split(representatives, doc["author_classes"])
    record = {"schema": PROTOCOL["schema"], "phase": "preflight_complete", "source_manifest": str(manifest), "source_manifest_sha256": sha256(manifest), "source_authority": doc["source_authority"], "protocol_path": str(protocol_path), "protocol_sha256": sha256(protocol_path), "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha256(__file__), "author_classes": doc["author_classes"], "source_record_count": len(source), "unique_source_paths": unique_paths, "exact_representatives_including_quarantine": len(representatives), "quarantined_proxy_groups": sorted(conflicts), "split_summary": summary, "scope": PROTOCOL["scope"], "source_caveat": PROTOCOL["source_caveat"]}
    with threadpool_limits(limits=1):
        record["self_test"], record["runtime"] = self_test(rows, groups, tr, dv), runtime_record()
        if args.check_only or args.self_test:
            print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
            return
        if args.output is None:
            raise ValueError("--output required for fitting")
        output = f_path(args.output, True)
        if output.exists():
            raise FileExistsError("Refusing overwrite or silent resume of existing run")
        output.mkdir(parents=True, exist_ok=False)
        start = time.monotonic()
        try:
            for key in ("TEMP", "TMP", "XDG_CACHE_HOME", "JOBLIB_TEMP_FOLDER"):
                f_path(os.environ[key], True).mkdir(parents=True, exist_ok=True)
            for row in mappings:
                row["split"] = representatives[row["representative_id"]]["split"]
            write_json(output / "frozen_split_manifest.json", {**record, "phase": "split_frozen_before_fit", "representatives": representatives, "all_source_mappings": mappings})
            write_json(output / "protocol.json", PROTOCOL)
            record["split_manifest_sha256"] = sha256(output / "frozen_split_manifest.json")
            matrices = {v["name"]: np.empty((len(rows), 71), dtype=np.float64) for v in PROTOCOL["variants"]}
            for i, row in enumerate(rows):
                for name, vector in describe(load_rgb(row["image_path"])).items():
                    matrices[name][i] = vector
                if (i + 1) % 500 == 0:
                    print(json.dumps({"phase": "features", "representatives": i + 1, "total": len(rows)}), flush=True)
            np.savez_compressed(output / "features.npz", **matrices, y=y, groups=groups, train_indices=tr, development_indices=dv, representative_ids=np.asarray([r["representative_id"] for r in rows]))
            write_json(output / "feature_schema.json", {"names": feature_names(), "regions": [{"name": name, "color_pixels": int(mask.sum()), "texture_pixels": int(texture.sum())} for name, mask, texture in regions()], "features_sha256": sha256(output / "features.npz")})
            results = {}
            for variant in PROTOCOL["variants"]:
                name, X = variant["name"], matrices[variant["name"]]
                scaler = StandardScaler()
                xtr = scaler.fit_transform(X[tr])
                model = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", class_weight="balanced", max_iter=2000, tol=1e-6, random_state=None)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(xtr, y[tr])
                if not np.array_equal(model.classes_, np.arange(4)) or not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
                    raise ValueError("Invalid fitted state")
                probabilities = model.predict_proba(scaler.transform(X))
                result = {"variant": variant, "fits": 1, "random_replicate_mean": None, "random_replicate_std": None, "n_iter": model.n_iter_.tolist(), "train": evaluate(y[tr], probabilities[tr], groups[tr], doc["author_classes"]), "development": evaluate(y[dv], probabilities[dv], groups[dv], doc["author_classes"])}
                np.savez_compressed(output / f"{name}_model.npz", coef=model.coef_, intercept=model.intercept_, classes=model.classes_, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, scaler_var=scaler.var_, n_samples_seen=np.asarray(scaler.n_samples_seen_), n_iter=model.n_iter_)
                with np.load(output / f"{name}_model.npz", allow_pickle=False) as saved:
                    logits = ((X[dv] - saved["scaler_mean"]) / saved["scaler_scale"]) @ saved["coef"].T + saved["intercept"]
                logits -= logits.max(axis=1, keepdims=True)
                reconstructed = np.exp(logits)
                reconstructed /= reconstructed.sum(axis=1, keepdims=True)
                if not np.allclose(reconstructed, probabilities[dv], rtol=1e-10, atol=1e-12):
                    raise AssertionError("Coefficient inference contract mismatch")
                result["saved_model_sha256"] = sha256(output / f"{name}_model.npz")
                write_json(output / f"{name}_metrics.json", result)
                write_json(output / f"{name}_predictions.json", {"class_order": doc["author_classes"], "rows": [{"representative_id": r["representative_id"], "image_path": r["image_path"], "author_class": r["author_class"], "proxy_group_id": r["recommended_proxy_group_id"], "split": r["split"], "predicted_author_class": doc["author_classes"][int(probabilities[i].argmax())], "probabilities": probabilities[i].tolist()} for i, r in enumerate(rows)]})
                results[name] = result
                print(json.dumps({"phase": "fit_complete", "variant": name, "development": result["development"]}, ensure_ascii=False), flush=True)
            comparison = {name: {f"development_{metric}_minus_whole": results[name]["development"][metric] - results["whole"]["development"][metric] for metric in ("accuracy", "balanced_accuracy", "macro_f1", "mean_proxy_group_accuracy")} for name in ("peripheral_8", "peripheral_16")}
            verify_sources(source)
            if sha256(manifest) != record["source_manifest_sha256"] or sha256(protocol_path) != record["protocol_sha256"] or sha256(__file__) != record["runner_sha256"] or sha256(output / "frozen_split_manifest.json") != record["split_manifest_sha256"]:
                raise ValueError("Frozen inputs changed during run")
            record.update(phase="complete", results=results, descriptive_comparison=comparison, source_bytes_and_frozen_inputs_rechecked_at_completion=True, elapsed_cpu_wall_seconds=time.monotonic() - start, interpretation_limit=PROTOCOL["evaluation"]["interpretation_limit"])
            write_json(output / "diagnostic_receipt.json", record)
            print(json.dumps({"phase": "complete", "receipt": str(output / "diagnostic_receipt.json"), "descriptive_comparison": comparison}, ensure_ascii=False), flush=True)
        except BaseException as exc:
            write_json(output / "FAILED.json", {**record, "phase": "failed", "exception": repr(exc), "traceback": traceback.format_exc(), "elapsed_cpu_wall_seconds": time.monotonic() - start, "partial_results_must_not_be_treated_as_success": True})
            raise


if __name__ == "__main__":
    main()
