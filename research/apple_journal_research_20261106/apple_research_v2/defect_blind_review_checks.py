"""Validate the immutable blind packet, then exercise GUI controls in a disposable copy."""

import csv
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import tkinter as tk
from unittest.mock import patch

from class0_patch_refinement import read, sha, atomic_json
from defect_blind_review_pack import OUT, MANIFEST, ROOT
from defect_blind_review_app import ResponseStore, Reviewer, OPTIONS


def check():
    public = OUT / "盲审材料"
    private = OUT / "组织者留存"
    record = read(private / "source_key.json")
    src = {r["image_id"]: r for r in read(MANIFEST)["samples"]}
    manifest = read(public / "cases.json")
    assert set(manifest) == {"schema", "cases"}
    native = {r["id"]: r for r in record["cases"]}
    raw_hashes = set()
    times = set()
    cases = manifest["cases"]
    assert (
        len(cases) == len(native) == 34
        and sha(public / "cases.json") == record["case_manifest_sha"]
    )
    assert set(c["id"] for c in cases) == native.keys()
    assert sha(public / "review.py") == sha(
        Path(__file__).with_name("defect_blind_review_app.py")
    )
    cmd = (public / "开始盲审.cmd").read_text("ascii")
    executable = cmd.split('start "" "%~dp0')[1].split('"')[0]
    assert (public / executable).resolve() == (
        ROOT.parent / "apple_detection/envs/local_rgbd_v1/Scripts/pythonw.exe"
    ).resolve()
    for case in cases:
        assert set(case) == {"id", "whole", "local", "center", "local_center"}
        key = native[case["id"]]
        source = src[key["image"]]
        assert (
            source["partition"] == "train"
            and sha(Path(source["rgb_path"])) == key["raw_sha"] == source["rgb_sha256"]
        )
        assert key["raw_sha"] not in raw_hashes and key["capture_group"] not in times
        raw_hashes.add(key["raw_sha"])
        times.add(key["capture_group"])
        assert sha(Path(source["label_path"])) == source["label_sha256"]
        with Image.open(source["rgb_path"]) as s, Image.open(
            public / case["whole"]
        ) as w, Image.open(public / case["local"]) as c:
            raw = np.asarray(s.convert("RGB"))
            assert np.array_equal(raw, np.asarray(w)) and c.size == (256, 256)
            l, t, r, b = key["crop_xyxy"]
            assert np.array_equal(raw[t:b, l:r], np.asarray(c)) and np.array_equal(
                np.subtract(case["center"], [l, t]), case["local_center"]
            )
            assert not w.getexif() and not c.getexif() and w.info == {} and c.info == {}
    assert raw_hashes.isdisjoint(
        {r["rgb_sha256"] for r in src.values() if r["partition"] != "train"}
    )
    original_gt_sha = sha(MANIFEST)
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        demo = Path(temp)
        shutil.copy(public / "cases.json", demo)
        # Use immutable image hard links; no original image or source GT can be written by the UI.
        shutil.copytree(public / "images", demo / "images", copy_function=os.link)
        s = ResponseStore(demo, "QA1")
        root = tk.Tk()
        root.withdraw()
        try:
            try:
                ResponseStore(demo, "QA1")
            except OSError:
                pass
            else:
                raise ValueError("Concurrent reviewer session was accepted")
            gui = Reviewer(root, s)
            for i, option in enumerate(OPTIONS):
                assert gui.index == i
                gui.radios[option].invoke()
                gui.next_blank()
            gui.move(-3)
            gui.radios["uncertain"].invoke()
            gui.zoom.set(True)
            gui.show()
            assert len(s.answers) == 4 and s.answers[cases[1]["id"]] == "uncertain"
            with patch("defect_blind_review_app.messagebox.showinfo") as success:
                gui.export()
                success.assert_called_once()
            exported = json.loads(
                next((demo / "导出结果").glob("*.json")).read_text("utf8")
            )
            with next((demo / "导出结果").glob("*.csv")).open(
                encoding="utf-8-sig", newline=""
            ) as f:
                csv_rows = list(csv.DictReader(f))
            assert (
                len(csv_rows) == 34 and sum(bool(r["response"]) for r in csv_rows) == 4
            )
            assert (
                exported["case_manifest_sha256"] == record["case_manifest_sha"]
                and exported["answered"] == 4
            )
            assert set(exported["cases"][0]) == {"id", "response", "判读"}
            expected = dict(s.answers)
            s.close()
            s = ResponseStore(demo, "QA1")
            assert s.answers == expected
            other = ResponseStore(demo, "QA2")
            assert not other.answers
            other.close()
        finally:
            s.close()
            root.destroy()
    assert sha(MANIFEST) == original_gt_sha
    values = [r for r in native.values() if r["source"] == "unmatched OOF candidate"]
    controls = [r for r in native.values() if r["source"] == "native positive"]
    refs = {(r["image"], r["pred_index"]): r for r in values}
    for r in controls:
        anchor = refs[tuple(r["matched_candidate"])]
        assert r["camera"] == anchor["camera"] and r["cls"] == anchor["cls"]
        assert all(1 / 1.30 <= v <= 1.30 for v in r["relative_size"])
    candidate_bounds = {
        str(cls): [
            min(r["confidence"] for r in values if r["cls"] == cls),
            max(r["confidence"] for r in values if r["cls"] == cls),
        ]
        for cls in (0, 1)
    }
    summary = dict(
        cases=len(cases),
        all_original_pixels_checked=True,
        train_unique_images=len(raw_hashes),
        train_unique_capture_seconds=len(times),
        development_or_quarantined_images=0,
        unmatched_class_counts=dict(Counter(r["cls"] for r in values)),
        control_class_counts=dict(Counter(r["cls"] for r in controls)),
        unmatched_score_range=candidate_bounds,
        control_max_absolute_axis_scale_ratio=max(
            max(float(v), 1 / float(v)) for r in controls for v in r["relative_size"]
        ),
        same_camera_all_controls=True,
        original_manifest_sha=original_gt_sha,
        case_sha=record["case_manifest_sha"],
        gui_test="withdrawn real Tk widgets, tested four radio clicks, editing, paging, zoom, autosave, locking, restart, independent reviewer, CSV/JSON export to disposable F folder",
    )
    atomic_json(private / "packet_audit.json", summary)
    print(summary)


if __name__ == "__main__":
    check()
