"""Reconcile saved blind decisions with the original, held-out training detections."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from audit_defect_causes_20261112 import match
from class0_patch_refinement import ROOT, atomic_json, read, sha
from defect_blind_review_app import OPTIONS, write_file
from defect_blind_review_pack import MANIFEST, OOF, OUT, native_gt, real_oof


NAMES = ["划伤", "虫伤"]
FINDINGS = {
    "missing_same_class": "人工支持漏标，预测类别相符",
    "missing_and_class_error": "人工支持漏标，预测类别有误",
    "false_detection": "目标缺陷误检",
    "uncertain": "人工无法判断",
    "unanswered": "未填写",
    "confirmed_native": "人工与原标一致",
    "disputed_native_class": "原标类别与人工不一致",
    "disputed_native_positive": "原标缺陷与人工不一致",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_export(path, packet, key, public):
    src = read(path)
    ids = [r["id"] for r in packet["cases"]]
    require(len(ids) == len(set(ids)) == src["total"] == 34, "Wrong packet size")
    require(
        src["schema"] == 1
        and src["case_manifest_sha256"]
        == key["case_manifest_sha"]
        == sha(public / "cases.json"),
        "Changed packet",
    )
    rows = src["cases"]
    require([r["id"] for r in rows] == ids, "Export links or ordering differ")
    for row in rows:
        answer = row["response"]
        require(answer in OPTIONS or answer == "", "Unknown answer")
        require(row["判读"] == OPTIONS.get(answer, "未填写"), "Different answer texts")
    answers = {r["id"]: r["response"] for r in rows if r["response"]}
    require(src["answered"] == len(answers), "Incorrect answered count")
    with path.with_suffix(".csv").open(encoding="utf-8-sig", newline="") as f:
        require(list(csv.DictReader(f)) == rows, "CSV and JSON disagree")
    # Autosave may legitimately be newer than a historical export. Compare but
    # never fill a blank export field from a non-exported decision.
    saves = [read(p) for p in (public / "本地填写").glob("*.json")]
    own = [
        s
        for s in saves
        if s["reviewer"] == src["reviewer"]
        and s["case_manifest_sha256"] == src["case_manifest_sha256"]
    ]
    require(len(own) <= 1, "Ambiguous autosave")
    return rows, own[0]["answers"] == answers if own else None


def interpret(kind, cls, manual):
    if manual in ("", "uncertain"):
        return "uncertain" if manual else "unanswered"
    same = manual == ("scratch", "insect")[cls]
    if kind == "control":
        if manual == "other":
            return "disputed_native_positive"
        return "confirmed_native" if same else "disputed_native_class"
    if manual == "other":
        return "false_detection"
    return "missing_same_class" if same else "missing_and_class_error"


def reconcile(path):
    public = OUT / "盲审材料"
    private = OUT / "组织者留存"
    key = read(private / "source_key.json")
    packet = read(public / "cases.json")
    responses, save_matches = load_export(path, packet, key, public)
    train, candidates, controls, folds = real_oof()
    require(
        key["ssda_manifest_sha"] == sha(MANIFEST) and key["folds"] == folds,
        "Original GT or OOF sources changed",
    )
    key_rows = {r["id"]: r for r in key["cases"]}
    require(
        len(key_rows) == 34 and set(key_rows) == {r["id"] for r in responses},
        "Incorrect organizer mapping",
    )
    originals = {("unmatched", r["image"], r["pred_index"]): r for r in candidates}
    originals.update({("control", r["image"], r["gt_index"]): r for r in controls})
    held = {
        r["image_id"]: r
        for fold in range(3)
        for r in read(OOF / f"fold{fold}" / "predictions.json")["rows"]
    }
    rows = []
    for n, (case, answer) in enumerate(zip(packet["cases"], responses, strict=True), 1):
        entry = key_rows[case["id"]]
        kind = "control" if entry["source"] == "native positive" else "unmatched"
        index = entry["gt_index" if kind == "control" else "pred_index"]
        origin = originals[(kind, entry["image"], index)]
        rgb = train[entry["image"]]
        require(
            all(origin[k] == entry[k] for k in origin), "Source row no longer matches"
        )
        require(
            rgb["partition"] == "train"
            and sha(Path(rgb["rgb_path"])) == rgb["rgb_sha256"] == entry["raw_sha"],
            "Changed source frame",
        )
        require(
            sha(Path(rgb["label_path"])) == rgb["label_sha256"]
            and rgb["same_second_group"] == entry["capture_group"],
            "Changed source labels or capture",
        )
        l, t, r, b = entry["crop_xyxy"]
        x1, y1, x2, y2 = origin["box"]
        expected_center = [round((x1 + x2) / 2), round((y1 + y2) / 2)]
        require(
            case["center"] == expected_center
            and case["local_center"]
            == [expected_center[0] - l, expected_center[1] - t],
            "Changed prompt location",
        )
        with (
            Image.open(rgb["rgb_path"]) as im,
            Image.open(public / case["whole"]) as whole,
            Image.open(public / case["local"]) as local,
        ):
            raw = np.asarray(im.convert("RGB"))
            require(
                np.array_equal(np.asarray(whole), raw)
                and np.array_equal(np.asarray(local), raw[t:b, l:r]),
                "Rendered pixels differ",
            )
        gt = native_gt(rgb)
        pred = held[entry["image"]]["pred"]
        pairs, overlap = match(pred, gt)
        if kind == "control":
            paired = next((pi for pi, gi in pairs.items() if gi == index), None)
            assessed = pred[paired] if paired is not None else None
            max_gt_iou = None
        else:
            max_gt_iou = float(max(overlap[index], default=0.0))
            require(
                index not in pairs and max_gt_iou < 0.1,
                "Candidate overlaps original labels",
            )
            assessed = pred[index]
        manual = answer["response"]
        rows.append(
            dict(
                position=n,
                id=case["id"],
                image_id=entry["image"],
                fold=entry["fold"],
                box_xyxy=entry["box"],
                center_xy=expected_center,
                kind=kind,
                reference_class=entry["cls"],
                reference_index=index,
                manual=manual,
                interpretation=interpret(kind, entry["cls"], manual),
                nearest_native_iou=max_gt_iou,
                oof_prediction=assessed,
            )
        )
    require(
        len({r["image_id"] for r in rows})
        == len({entry["capture_group"] for entry in key_rows.values()})
        == 34,
        "Repeated views",
    )
    cross = Counter((r["kind"], r["reference_class"], r["manual"]) for r in rows)
    return dict(
        schema=1,
        cases=34,
        answered=sum(bool(r["manual"]) for r in rows),
        csv_identical=True,
        autosave_matches_export=save_matches,
        export_sha256=sha(path),
        csv_sha256=sha(path.with_suffix(".csv")),
        manifest_sha256=sha(MANIFEST),
        packet_sha256=key["case_manifest_sha"],
        fold_provenance=folds,
        source_key_sha256=sha(private / "source_key.json"),
        counts=dict(Counter(r["interpretation"] for r in rows)),
        cross_table=[
            dict(kind=k, reference_class=c, manual=m, n=v)
            for (k, c, m), v in sorted(cross.items())
        ],
        rows=rows,
    )


def apply_supplement(audit, path):
    """Attach an explicit user answer to an unanswered item without editing the export."""
    note = read(path)
    require(
        note["source"] == "user_chat"
        and note["export_sha256"] == audit["export_sha256"],
        "Supplement has the wrong source or export",
    )
    row = next((r for r in audit["rows"] if r["id"] == note["id"]), None)
    require(
        row is not None and row["position"] == note["position"] and not row["manual"],
        "Supplement must refer to a blank item",
    )
    answer = note["response"]
    aliases = {"uncertain": ("看不清",), "other": ("其他", "普通果皮")}
    require(
        answer in OPTIONS
        and note["original_reply"] in (OPTIONS[answer], *aliases.get(answer, ())),
        "Invalid supplemental judgement",
    )
    audit["export_answered"] = audit["answered"]
    row.update(
        manual=answer,
        manual_source="user_chat",
        interpretation=interpret(row["kind"], row["reference_class"], answer),
    )
    audit["answered"] += 1
    audit["supplement"] = dict(
        id=row["id"], position=row["position"], source="user_chat", sha256=sha(path)
    )
    audit["counts"] = dict(Counter(r["interpretation"] for r in audit["rows"]))
    cross = Counter(
        (r["kind"], r["reference_class"], r["manual"]) for r in audit["rows"]
    )
    audit["cross_table"] = [
        dict(kind=k, reference_class=c, manual=m, n=v)
        for (k, c, m), v in sorted(cross.items())
    ]
    return audit


def write_report(audit, out):
    answered = audit.get("export_answered", audit["answered"])
    note = ""
    if "supplement" in audit:
        note = f"第 {audit['supplement']['position']} 处由用户在对话中补答，合计 {audit['answered']} 处已判读。"
    lines = [
        "# SSDA 训练图 34 处人工盲审核对",
        "",
        f"原始导出共 34 条，其中 {answered} 条有答案。CSV 与 JSON 完全一致。{note}样本来自 34 张不同训练图与 34 个不同采集秒。已核验原图、裁片、定位点、原标签与折外权重及预测的校验值。",
        "",
        "24 处取自原标注范围以外的高分折外预测（各预测类别 12 处），10 处为相机和尺寸匹配的原标注对照（各类别 5 处）。每处由一份人工答案判读定位点中心，未绘制缺陷边界。判为“其他／普通果皮”可确认该位置不支持目标缺陷，但并未细分其他病害。困难例经刻意筛选，不能据此推算整个数据集的漏标率。",
        "",
    ]
    counts = Counter(row["interpretation"] for row in audit["rows"])
    lines.extend(
        [
            f"原标以外的 24 个候选中，{counts['missing_same_class']} 处与预测类别相符，{counts['missing_and_class_error']} 处有缺陷但预测类别有误，{counts['false_detection']} 处被判为其他／普通果皮。原标注对照中 {counts['confirmed_native']}/10 处一致。",
            "",
            "| 编号 | 参考来源 | 参考类 | 人工判读 | 归因 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in audit["rows"]:
        if not row["manual"]:
            lines.append(
                f"| {row['position']:02d} | — | — | 未填写 | 单独保留，补录后揭盲 |"
            )
            continue
        source = "原标对照" if row["kind"] == "control" else "未匹配预测"
        note = "（对话补答）" if row.get("manual_source") == "user_chat" else ""
        lines.append(
            f"| {row['position']:02d} | {source} | {NAMES[row['reference_class']]} | {OPTIONS.get(row['manual'], '未填写')}{note} | {FINDINGS[row['interpretation']]} |"
        )
    lines += [
        "",
        "## 训练用途",
        "",
        "未匹配位置中，人工判为划伤、虫伤或其他的样本可用作中心区域的弱分类约束。来源框由检测器生成，没有人工核定的边界，不能直接充当框回归真值。对原标注存在分歧的样本保留原始记录，可交第二位审阅人独立判读。未填写和无法判断的条目不参与弱监督。开发集的原始标注与既有评价口径保持一致。",
        "",
    ]
    write_file(out / "逐例核对.md", "\n".join(lines).encode("utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="Explicit review JSON export")
    parser.add_argument(
        "--supplement", type=Path, help="A separately sourced answer for a blank item"
    )
    args = parser.parse_args()
    files = sorted((OUT / "盲审材料" / "导出结果").glob("*.json"))
    if args.export:
        path = args.export
    else:
        require(
            bool(files) and len({read(f)["reviewer"] for f in files}) == 1,
            "Choose one independent reviewer's export with --export",
        )
        path = files[-1]
    audit = reconcile(path)
    out = OUT / "核对结果"
    if args.supplement:
        audit = apply_supplement(audit, args.supplement)
        out = out / "34处最终核对"
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "cases_audit.json", audit)
    write_report(audit, out)
    print(
        f"Validated {audit['answered']}/34 answers, counts={audit['counts']}, autosave agrees={audit['autosave_matches_export']}"
    )


if __name__ == "__main__":
    main()
