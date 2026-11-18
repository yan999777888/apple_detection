"""Download the author FruitLife apple trajectories and isolate the documented fruit ROI."""
import csv
import hashlib
import io
import json
import os
import tarfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets/public_defect_intake/fruitlife_v1"
RECORD = "23589223"
ARCHIVE = "fruitlife_main_images_v1.0.tar.gz"
MASKS = "fruitlife_masks_v1.3_shadowfree.tar.gz"
LABELS = "manifest_v2_b6consensus.csv"
BLOCK = 8 * 1024 ** 2


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_bytes(data)
    os.replace(temp, path)


def get(url, headers=None):
    for attempt in range(9):
        try:
            response = requests.get(url, headers=headers, timeout=(10, 60))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 8:
                raise
            time.sleep(min(6, attempt + 1))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_files():
    meta = OUT / "zenodo_record.json"
    if not meta.exists():
        save(meta, get(f"https://zenodo.org/api/records/{RECORD}").content)
    record = json.loads(meta.read_bytes())
    if str(record["id"]) != RECORD or record["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("Record identifier or declared license changed")
    entries = {e["key"]: e for e in record["files"]}
    for name in (LABELS, MASKS, "README.md", "CHECKSUMS.sha256"):
        item = entries[name]
        target = OUT / name
        if not target.exists():
            save(target, get(item["links"]["self"]).content)
        if "md5:" + hashlib.md5(target.read_bytes()).hexdigest() != item["checksum"]:
            raise ValueError(f"Author file checksum mismatch for {name}")
    return entries[ARCHIVE]


def read_block(index, total):
    start = index * BLOCK
    stop = min(start + BLOCK, total) - 1
    target = OUT / "archive_chunks" / f"{index:04d}.partdata"
    if not target.exists():
        url = f"https://zenodo.org/records/{RECORD}/files/{ARCHIVE}?download=1&block={start}"
        r = get(url, {"Range": f"bytes={start}-{stop}", "Accept-Encoding": "identity"})
        if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{stop}/{total}":
            raise ValueError(f"Archive range {index} returned a different range")
        if len(r.content) != stop - start + 1:
            raise ValueError(f"Archive range {index} truncated")
        save(target, r.content)
    data = target.read_bytes()
    if len(data) != stop - start + 1:
        raise ValueError(f"Stored archive range {index} truncated")
    return data


class RangeReader:
    def __init__(self, total, pool):
        self.total, self.pool = total, pool
        self.blocks = (total + BLOCK - 1) // BLOCK
        self.queue = {}
        self.index = 0
        self.buffer = io.BytesIO()
        self.prefetch()

    def prefetch(self):
        for k in range(self.index, min(self.blocks, self.index + 8)):
            if k not in self.queue:
                self.queue[k] = self.pool.submit(read_block, k, self.total)

    def read(self, size):
        data = self.buffer.read(size)
        if data or self.index == self.blocks:
            return data
        self.buffer = io.BytesIO(self.queue.pop(self.index).result())
        self.index += 1
        self.prefetch()
        return self.buffer.read(size)


def fetch_apples(archive, rows):
    wanted = {f"data_clean/APPLE/{r['specimen_id']}/{r['filename']}" for r in rows}
    rgb = OUT / "source"
    if all((rgb / key).is_file() for key in wanted):
        return
    with ThreadPoolExecutor(max_workers=6) as pool:
        stream = RangeReader(archive["size"], pool)
        with tarfile.open(fileobj=stream, mode="r|gz", bufsize=1024 ** 2) as tf:
            for member in tf:
                if not member.isfile() or member.name not in wanted:
                    continue
                data = tf.extractfile(member).read()
                if len(data) != member.size or not data.startswith(b"\xff\xd8"):
                    raise ValueError(f"Invalid archived JPEG: {member.name}")
                save(rgb / member.name, data)
                wanted.remove(member.name)
                if len(wanted) % 60 == 0:
                    print(f"Apple JPEGs extracted: {len(rows)-len(wanted)} / {len(rows)}", flush=True)
                if not wanted:
                    for future in stream.queue.values():
                        future.cancel()
                    break
        if wanted:
            raise ValueError(f"Missing {len(wanted)} author images")


def project(rows):
    cv2.setNumThreads(1)
    result = []
    with tarfile.open(OUT / MASKS) as t:
        for r in rows:
            fruit, name = r["specimen_id"], r["filename"]
            orig = OUT / "source/data_clean/APPLE" / fruit / name
            raw = orig.read_bytes()
            # Author masks were built from the stored raster before smartphone EXIF rotation.
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            mask_bytes = t.extractfile(f"masks/APPLE/{fruit}/{Path(name).stem}.png").read()
            mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None or bgr is None or not mask.any():
                raise ValueError(f"Invalid apple pixels or author mask: {name}")
            h, w = bgr.shape[:2]
            mh, mw = mask.shape
            if abs(h / w - mh / mw) > .014:
                raise ValueError(f"Source image orientation and mask mismatch: {name}")
            x, y, bw, bh = cv2.boundingRect(mask)
            # Author masks are in a downscaled grid; preserve aspect and the 10% context used in deployment.
            l, rgt = max(0, int((x - .1*bw)*w/mw)), min(w, int(np.ceil((x+1.1*bw)*w/mw)))
            top, bottom = max(0, int((y - .1*bh)*h/mh)), min(h, int(np.ceil((y+1.1*bh)*h/mh)))
            # Images contain a ruler and a colour reference card. Restrict visible
            # pixels to the author's shadow-free mask in every capture condition.
            fruit_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)[top:bottom,l:rgt]
            crop = bgr[top:bottom,l:rgt].copy()
            crop[fruit_mask == 0] = 114
            if crop.size == 0:
                raise ValueError(f"Empty apple crop: {name}")
            ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise ValueError(f"Cannot encode crop {name}")
            out = OUT / "crops" / fruit / name
            save(out, jpg.tobytes())
            result.append(dict(path=out.relative_to(ROOT).as_posix(), source_path=orig.relative_to(ROOT).as_posix(),
                               sha256=sha(jpg.tobytes()), source_sha256=sha(raw), author_mask_sha256=sha(mask_bytes),
                               group_id=fruit, specimen_id=fruit, storage=r["storage"], author_stage=int(r["stage"]),
                               source_stage=int(r["stage"]), time=r["timestamp"], crop_path=out.relative_to(ROOT).as_posix(),
                               stage_original=int(r["stage_original"]), blind_relabeled=r["b6_relabeled"].lower()=="true",
                               xyxy=[l,top,rgt,bottom], source_hw=[h,w],
                               crop_method="author shadow-free fruit mask; outside pixels gray 114; bounding box plus 10%"))
    if len({r["source_sha256"] for r in result}) != len(result):
        raise ValueError("There are exact duplicate apple sources; audit before use.")
    save(OUT / "apple_manifest.json", json.dumps({"samples":result},indent=2).encode())
    return result


if __name__ == "__main__":
    archive = source_files()
    with (OUT / LABELS).open(encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["class"] == "apple"]
    if len(rows) != 360 or len({r["specimen_id"] for r in rows}) != 6:
        raise ValueError("Author apple population changed")
    fetch_apples(archive, rows)
    results = project(rows)
    print("Author stages:", dict(Counter(r["author_stage"] for r in results)), flush=True)
