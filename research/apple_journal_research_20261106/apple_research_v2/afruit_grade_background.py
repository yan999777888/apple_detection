"""Paired train-only background randomization. Both arms see identical held-out inputs."""
import argparse
import collections
import json
import shutil

import cv2
import numpy as np
import torch

import afruit_grade_model as grade
import afruit_grade_train as paired


OUT = grade.ROOT / "experiments/afruit_grade_background_v1"
RELEASE = grade.ROOT / "experiments/afruit_grade_release_v1"
BG_SEED = 11827
PROB = .85


def fruit_mask(rgb):
    """Conservative connected colored fruit envelope; fill holes to protect dark defects."""
    side = grade.SIDE
    smooth = cv2.GaussianBlur(rgb, (9, 9), 0)
    u = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
    sat = u[:, :, 1]
    cut, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    r, g, b = smooth.astype(np.float32).transpose(2, 0, 1)
    seeds = ((sat > np.clip(cut * .76, 40, 83)) & (r > b*.93) & (g > b*.65) & (u[:,:,2] > 24)).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, kernel)
    cc, _ = cv2.findContours(seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yy, xx = np.indices((side, side), dtype=np.float32)
    spatial = np.exp(-((yy-side/2)**2+(xx-side/2)**2)/(2*(side*.26)**2))
    options = []
    for contour in cc:
        area = cv2.contourArea(contour)
        if area < side*side*.008:
            continue
        hull = cv2.convexHull(contour)
        mask = cv2.drawContours(np.zeros_like(seeds), [hull], -1, 1, -1)
        harea = mask.sum()
        options.append((float(spatial[mask>0].sum()*(area/max(harea,1))**.5), mask))
    if not options:
        # No trustworthy training foreground: the image must be left untouched.
        return None
    mask = max(options, key=lambda el: el[0])[1]
    if mask.mean()<.012 or mask.mean()>.84:
        return None
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    return mask


def clean_donor(rgb, mask):
    # Replace the entire potential donor fruit before borrowing its backdrop.
    small = cv2.resize(rgb, (56,56), interpolation=cv2.INTER_AREA)
    tiny = cv2.resize(mask, (56,56), interpolation=cv2.INTER_AREA)
    blank = cv2.dilate(tiny, np.ones((3,3),np.uint8))
    clean = cv2.inpaint(small, blank, 4, cv2.INPAINT_TELEA)
    restored = cv2.resize(clean, (grade.SIDE,grade.SIDE), interpolation=cv2.INTER_LINEAR)
    out = rgb.copy()
    region = cv2.dilate(mask, np.ones((7,7), np.uint8)).astype(bool)
    out[region] = restored[region]
    return out


class Backgrounds:
    def __init__(self, records, images):
        self.rng = np.random.default_rng(BG_SEED)
        self.images = images
        self.ids = {id(im):i for i,im in enumerate(images)}
        self.masks = [fruit_mask(im) for im in images]
        self.donors = [None if m is None else clean_donor(im,m) for im,m in zip(images,self.masks)]
        self.by_grade = [[j for j,r in enumerate(records) if r["grade"]==g and self.donors[j] is not None] for g in range(3)]
        if min(map(len,self.by_grade))==0:
            raise ValueError("No valid train-only donors for one grade")
        self.records = records
        self.replaced = 0
        self.pair_counts = collections.Counter()
        self.alphas = []
        for mask in self.masks:
            if mask is None:
                self.alphas.append(None)
            else:
                # Feather only outside the protected pixels, leaving every
                # original pixel inside the conservative envelope bit-identical.
                alpha = cv2.GaussianBlur(mask.astype(np.float32), (9,9), 1.8)
                alpha[mask>0] = 1.
                self.alphas.append(alpha[:,:,None])

    def swap(self, image, dry=False):
        idx = self.ids[id(image)]
        mask, alpha = self.masks[idx], self.alphas[idx]
        if mask is None or (not dry and self.rng.random()>=PROB):
            return image
        donor_grade = int(self.rng.integers(3))
        candidates = [j for j in self.by_grade[donor_grade] if self.records[j]["group_id"] != self.records[idx]["group_id"]]
        if not candidates:
            return image
        d = int(self.rng.choice(candidates))
        out = (alpha*image + (1-alpha)*self.donors[d]).round().astype(np.uint8)
        if dry:
            if not np.array_equal(out[mask>0], image[mask>0]):
                raise ValueError("Protected apple pixels changed")
        else:
            self.replaced += 1
            self.pair_counts[(self.records[idx]["grade"],donor_grade)] += 1
        return out


def freeze():
    protocol = dict(baseline_protocol=grade.file_sha(grade.OUT/"protocol.json"),
       manifest=grade.file_sha(paired.MANIFEST), source=grade.file_sha(__file__), seed=BG_SEED, probability=PROB,
       change="Before each otherwise identical training augmentation, preserve a convex connected, dilated fruit mask and borrow background from a different training capture group, with donor grade uniform and independent of source.",
       mask="HSV saturation threshold clipped to 40..83; smooth and close; choose largest central-weighted compact component; fill convex hull and dilate 6px. Fallback to original when invalid. Feather only outside protected envelope. Donor fruit is inpainted first.",
       unchanged="Validation/test and deployment preprocessing, ImageNet initialization, minibatch order, common augmentation RNG, learning rate, epoch and threshold selection kept fixed.",
       decision="Use train-only background arm iff on the same validation split medium correct increases by >=3, severe-to-good count does not increase, and macro recall improves. Compare selected operating point of each. Otherwise baseline.",
       test="Evaluate winner only, once, after immutable release decision. No test-based checkpoint selection.")
    dest = OUT/"protocol.json"
    if dest.exists() and paired.read(dest)!=protocol:
        raise ValueError("Background protocol already differs")
    paired.atomic(dest, protocol)


def preview():
    freeze()
    records, imgs, _ = paired.open_rows(("train",))["train"]
    bg = Backgrounds(records, imgs)
    rng = np.random.default_rng(BG_SEED)
    tiles = []
    for g in range(3):
        group = [i for i,r in enumerate(records) if r["grade"]==g]
        for i in rng.choice(group,8,replace=False):
            im = imgs[i]
            out = bg.swap(im, dry=True)
            mask = bg.masks[i]
            checked = im.copy()
            if mask is not None:
                contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(checked,contours,-1,(0,235,230),2)
            tiles += [cv2.hconcat([cv2.resize(c,(100,100)) for c in (checked,out)])]
    page = cv2.vconcat([cv2.hconcat(tiles[i:i+4]) for i in range(0,len(tiles),4)])
    cv2.imencode(".jpg",cv2.cvtColor(page,cv2.COLOR_RGB2BGR))[1].tofile(OUT/"train_mask_preview.jpg")
    vals = [float(m.mean()) for m in bg.masks if m is not None]
    paired.atomic(OUT/"preflight.json",dict(train=len(records),valid_mask=len(vals),fallback=len(records)-len(vals),
                  protected_fraction_quantiles=np.quantile(vals,[0,.1,.5,.9,1]).tolist(),preview_only_train=True))


def train():
    freeze()
    if not (OUT/"preflight.json").is_file():
        raise ValueError("Training-mask visual review is required before this registered comparison")
    old_read,old_aug = paired.open_rows,paired.augment
    paired.OUT = OUT
    paired.freeze = freeze
    state = {}
    def open_pair(splits):
        data = old_read(splits)
        state["bg"] = Backgrounds(*data["train"][:2])
        return data
    def changed(im,rng):
        return old_aug(state["bg"].swap(im), rng)
    paired.open_rows, paired.augment = open_pair, changed
    paired.train()
    bg = state["bg"]
    paired.atomic(OUT/"augmentation_receipt.json",dict(replaced=bg.replaced,pairs={str(k):v for k,v in bg.pair_counts.items()}))


def release():
    freeze()
    base = paired.read(grade.OUT/"selection.json")
    bg = paired.read(OUT/"selection.json")
    a,b = (s["validation_at_frozen_point"] for s in (base,bg))
    decision = dict(medium_gain=b["confusion"][1][1]-a["confusion"][1][1],
                    risk_delta=b["severe_to_good_count"]-a["severe_to_good_count"],
                    macro_recall_gain=b["macro_recall"]-a["macro_recall"])
    use_bg = decision["medium_gain"]>=3 and decision["risk_delta"]<=0 and decision["macro_recall_gain"]>0
    folder = OUT if use_bg else grade.OUT
    selected = paired.read(folder/"selection.json")
    if grade.file_sha(folder/"weights.pt")!=selected["weights_sha256"]:
        raise ValueError("Invalid selected weights")
    decision.update(source=folder.name, source_selection_sha=grade.file_sha(folder/"selection.json"),
                    baseline_sha=base["weights_sha256"],background_arm_sha=bg["weights_sha256"])
    if (RELEASE/"selection.json").exists():
        raise ValueError("Release already frozen")
    RELEASE.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(folder/"weights.pt", RELEASE/"weights.pt")
    shutil.copyfile(folder/"protocol.json", RELEASE/"protocol.json")
    paired.atomic(RELEASE/"decision.json",decision)
    paired.atomic(RELEASE/"selection.json",selected)
    print("Frozen release",decision,flush=True)


def final_test():
    import afruit_grade_release
    paired.OUT = RELEASE
    paired.grade.load_model = afruit_grade_release.load_model
    paired.test()


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["preview","train","release","test"])
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    {"preview":preview,"train":train,"release":release,"test":final_test}[ap.parse_args().action]()
