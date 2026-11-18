"""Fixed, validation-selected AFruitDB grading study on grouped apple images."""
import argparse
import collections
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import beta
from torch import nn

import afruit_grade_model as grade


ROOT, OUT = grade.ROOT, grade.OUT
MANIFEST = ROOT / "datasets/afruitdb_apples_v1/manifest.json"
INITIAL = ROOT / "runtime/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth"
SEED = 18112026
EPOCHS = 24
BATCH = 24
THRESHOLDS = (1.0, .30, .20, .12, .08)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic(path, obj, tensor=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    if tensor:
        torch.save(obj, tmp)
    else:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def freeze():
    source = grade.file_sha(INITIAL)
    if not source.startswith("047dcff4"):
        raise ValueError("Unexpected torchvision ImageNet-1K checkpoint")
    protocol = dict(seed=SEED, epochs=EPOCHS, batch=BATCH, side=grade.SIDE,
                    class_order=grade.LABELS, architecture="MobileNet V3 Small",
                    initial="torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1", pretrained_sha256=source,
                    optimizer="AdamW; backbone 0.0002, head 0.001; weight decay 0.008; cosine floor 0.10; fine-tune all",
                    loss="CE weighted by inverse sqrt train class frequencies, label smoothing 0.03",
                    augmentation="Horizontal flip, -10 to +10 degrees and mild brightness/contrast; train only",
                    checkpoints=list(range(1, EPOCHS + 1)),
                    checkpoint_policy="Max validation macro recall - 0.40*severe-to-good rate - 0.12*good-to-severe rate; argmax; then lower NLL, earlier epoch",
                    thresholds=list(THRESHOLDS),
                    threshold_policy="On selected checkpoint, allow good to middle if p(severe)>=cut. Validation good recall loss <=0.10 vs argmax. Max same utility; ties prefer fewer changes (higher cutoff).",
                    test_policy="One frozen pass. Report ordinary argmax and selected guard on the same logits. No test-time calibration.",
                    maturity="The labels and this model cover external appearance only. No joint maturity labels assumed.",
                    file_sha256={Path(__file__).name: grade.file_sha(__file__),
                                 "afruit_grade_model.py": grade.file_sha(Path(grade.__file__))})
    stored = OUT / "protocol.json"
    if stored.exists():
        if read(stored) != json.loads(json.dumps(protocol)):
            raise RuntimeError("Frozen protocol has changed")
    else:
        atomic(stored, protocol)


def open_rows(subsets):
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    rows = read(MANIFEST)["samples"]
    partition_for_group, partition_for_hash = {}, {}
    for r in rows:
        if int(r["grade"]) not in (0, 1, 2) or r["split"] not in ("train", "val", "test"):
            raise ValueError("Unknown class or split")
        for field, seen in (("group_id", partition_for_group), ("sha256", partition_for_hash)):
            if seen.setdefault(r[field], r["split"]) != r["split"]:
                raise ValueError(f"Repeated {field} across splits")
    result = {}
    for subset in subsets:
        group = [r for r in rows if r["split"] == subset]
        if set(int(r["grade"]) for r in group) != {0, 1, 2}:
            raise ValueError(f"Missing grade in {subset}")
        images = []
        for r in group:
            path = ROOT / r["path"]
            if grade.file_sha(path) != r["sha256"]:
                raise ValueError(f"Image changed: {path}")
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            images.append(grade.rgb_square(image))
        result[subset] = (group, images, np.array([int(r["grade"]) for r in group]))
    return result


def augment(im, rng):
    if rng.random() < .5:
        im = cv2.flip(im, 1)
    angle = rng.uniform(-10, 10)
    s = grade.SIDE
    matrix = cv2.getRotationMatrix2D((s / 2, s / 2), angle, 1.)
    im = cv2.warpAffine(im, matrix, (s, s), borderMode=cv2.BORDER_REFLECT_101)
    return np.clip(im.astype(np.float32)*rng.uniform(.90, 1.10) + rng.uniform(-8, 8), 0, 255).astype(np.uint8)


def metrics(target, result):
    truth, pred = np.asarray(target, int), np.asarray(result, int)
    matrix = np.bincount(truth * 3 + pred, minlength=9).reshape(3, 3)
    totals = matrix.sum(axis=1)
    recalled = matrix.diagonal() / totals
    predicted = matrix.sum(axis=0)
    precisions = np.divide(matrix.diagonal(), predicted, out=np.zeros(3, dtype=float), where=predicted > 0)
    bad_good, good_bad = matrix[2, 0]/totals[2], matrix[0, 2]/totals[0]
    return dict(n=len(target), confusion=matrix.tolist(), accuracy=float(matrix.trace()/len(target)),
                per_class=[dict(grade=grade.LABELS[i], n=int(totals[i]), predicted=int(predicted[i]),
                                precision=float(precisions[i]) if predicted[i] else None,
                                recall=float(recalled[i])) for i in range(3)],
                macro_recall=float(recalled.mean()), severe_to_good_count=int(matrix[2, 0]),
                severe_n=int(totals[2]), severe_to_good_rate=float(bad_good),
                good_to_severe_count=int(matrix[0, 2]), good_n=int(totals[0]),
                good_to_severe_rate=float(good_bad), ordinal_mae=float(np.abs(truth-pred).mean()),
                utility=float(recalled.mean() - .4*bad_good - .12*good_bad))


@torch.inference_mode()
def logits_for(model, images):
    model.eval()
    return torch.cat([model(grade.tensor_rgb(images[start:start+BATCH], "cuda")).float().cpu()
                      for start in range(0, len(images), BATCH)])


def choose(logits, truth):
    alternatives = []
    for threshold in THRESHOLDS:
        item = metrics(truth, grade.decide(logits, threshold))
        item["threshold"] = threshold
        alternatives.append(item)
    recall_floor = alternatives[0]["per_class"][0]["recall"] - .10
    eligible = [a for a in alternatives if a["per_class"][0]["recall"] + 1e-9 >= recall_floor]
    return max(eligible, key=lambda a: (round(a["utility"], 12), a["threshold"])), alternatives


def train():
    freeze()
    if (OUT/"selection.json").exists():
        raise ValueError("Training already finished; refusing to replace selected weights")
    data = open_rows(("train", "val"))
    rows, ims, ys = data["train"]
    vrows, vi, vy = data["val"]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = grade.network(pretrained=True).cuda()
    w = 1 / np.sqrt(np.bincount(ys))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w/w.mean(), dtype=torch.float32, device="cuda"), label_smoothing=.03)
    optimizer = torch.optim.AdamW([dict(params=model.features.parameters(), lr=.0002),
                                   dict(params=model.classifier.parameters(), lr=.001)], weight_decay=.008)
    batches = math.ceil(len(ys) / BATCH)
    lrs = [group["lr"] for group in optimizer.param_groups]
    history = []
    best, best_key, best_logits = None, None, None
    before = time.perf_counter()
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.
        # Repartition so the last batch never has only one image.
        partitions = np.array_split(rng.permutation(len(rows)), batches)
        for batch_idx, ids in enumerate(partitions):
            progress = ((epoch - 1) * batches + batch_idx) / max(1, EPOCHS*batches-1)
            gain = .1 + .9 * (1 + math.cos(math.pi * progress)) / 2
            for group, base in zip(optimizer.param_groups, lrs):
                group["lr"] = base * gain
            x = grade.tensor_rgb([augment(ims[i], rng) for i in ids], "cuda")
            target = torch.as_tensor(ys[ids], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach())*len(ids)
        z = logits_for(model, vi)
        stats = metrics(vy, z.argmax(dim=1))
        nll = nn.functional.cross_entropy(z, torch.as_tensor(vy)).item()
        key = (round(stats["utility"], 12), -nll, -epoch)
        if best_key is None or key > best_key:
            best, best_key, best_logits = epoch, key, z.clone()
            atomic(OUT/"weights.pt", model.state_dict(), tensor=True)
        history.append(dict(epoch=epoch, train_loss=total_loss/len(ys), validation_nll=nll,
                            validation=stats, elapsed_seconds=time.perf_counter()-before))
        atomic(OUT/"history.json", history)
        print(f"Epoch {epoch:02d} val {stats['confusion']} accuracy {stats['accuracy']:.3f} best {best}", flush=True)
    selected, candidates = choose(best_logits, vy)
    atomic(OUT/"selection.json", dict(epoch=best, threshold=selected["threshold"],
           weights_sha256=grade.file_sha(OUT/"weights.pt"), protocol_sha256=grade.file_sha(OUT/"protocol.json"),
           manifest_sha256=grade.file_sha(MANIFEST), train_n=len(rows), val_n=len(vrows),
           images_seen=sum(len(ys) for _ in history), updates=EPOCHS*batches, device=torch.cuda.get_device_name(),
           elapsed_seconds=time.perf_counter()-before, validation_at_frozen_point=selected, alternatives=candidates))


def risk_interval(events, opportunities):
    """Two-sided 95% exact binomial interval, with conventional endpoints at 0 and 1."""
    lower = 0. if not events else beta.ppf(.025, events, opportunities-events+1)
    upper = 1. if events == opportunities else beta.ppf(.975, events+1, opportunities-events)
    return [float(lower), float(upper)]


def test():
    selection = read(OUT/"selection.json")
    if (OUT/"test.json").exists():
        raise ValueError("A frozen test result already exists; no repeated selection/evaluation")
    if selection["manifest_sha256"] != grade.file_sha(MANIFEST) or selection["protocol_sha256"] != grade.file_sha(OUT/"protocol.json"):
        raise ValueError("Manifest or protocol changed")
    rows, ims, y = open_rows(("test",))["test"]
    model, cutoff = grade.load_model("cuda")
    logits = logits_for(model, ims)
    guarded = grade.decide(logits, cutoff).numpy()
    stats = metrics(y, guarded)
    bad_groups = collections.defaultdict(list)
    for r, output in zip(rows, guarded, strict=True):
        if r["grade"] == 2:
            bad_groups[r["group_id"]].append(int(output)==0)
    failed_groups = sum(any(flags) for flags in bad_groups.values())
    stats["severe_to_good_exact_95_if_images_independent"] = risk_interval(stats["severe_to_good_count"], stats["severe_n"])
    stats["severe_group_any_as_good"] = dict(groups=len(bad_groups), groups_with_any_error=failed_groups,
        exact_95_if_groups_independent=risk_interval(failed_groups, len(bad_groups)))
    stats["interval_scope"] = "Intervals conditional on approximate independence; audited near-duplicate groups may differ from real fruit identities."
    probabilities = logits.softmax(1).tolist()
    pred = [dict(path=r["path"], group_id=r["group_id"], sha256=r["sha256"], grade=int(actual),
                 predicted=int(pr), probabilities=prob) for r,actual,pr,prob in zip(rows,y,guarded,probabilities,strict=True)]
    atomic(OUT/"test_predictions.json", pred)
    atomic(OUT/"test.json", dict(source="AFruitDB apple image-level grades, held-out groups; one pass, no orchard or maturity truth",
        weighted_decision=stats, ordinary_argmax=metrics(y, logits.argmax(1)), threshold=cutoff,
        weights_sha256=selection["weights_sha256"], manifest_sha256=selection["manifest_sha256"],
        selection_sha256=grade.file_sha(OUT/"selection.json"), predictions_sha256=grade.file_sha(OUT/"test_predictions.json")))
    print(stats, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "train", "test"])
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    {"freeze":freeze, "train":train, "test":test}[parser.parse_args().action]()
