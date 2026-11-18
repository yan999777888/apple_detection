"""Build a local, read-only gallery linking each review decision to its original image."""

import argparse
from collections import Counter
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote

from PIL import Image


PACKET = Path(__file__).resolve().parents[1] / "experiments" / "ssda_blind_defects_v1"
DECISIONS = {"scratch": "划伤", "insect": "虫伤", "other": "其他／普通果皮",
             "uncertain": "看不清", "": "未填写"}
GROUPS = ("缺陷漏标线索，预测类别一致", "缺陷漏标线索，预测类别有误", "人工判为其他的误检",
          "无法确定或未填写", "原标对照一致", "原标对照有分歧")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def group(source, expected, answer):
    if answer in ("", "uncertain"):
        return GROUPS[3]
    if source == "native positive":
        return GROUPS[4] if expected == answer else GROUPS[5]
    if source != "unmatched OOF candidate":
        raise ValueError(f"Unknown source: {source}")
    if answer == "other":
        return GROUPS[2]
    return GROUPS[0] if expected == answer else GROUPS[1]


STYLE = """
*{box-sizing:border-box}body{margin:0 auto;padding:1.6rem;max-width:1250px;background:#fafaf7;color:#202a28;
font:16px/1.6 Inter,"Microsoft YaHei",sans-serif}a{color:#125c51}h1,h2{line-height:1.3}
nav{display:flex;flex-wrap:wrap;gap:.45rem 1.2rem}section{margin-top:2.4rem}section h2{font-size:1.25rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(195px,1fr));gap:1rem}
article{background:white;border:1px solid #cbd4cf;padding:.7rem;border-radius:6px}
.crop{position:relative;display:block;line-height:0}.crop img{width:100%;height:auto}
.mark{position:absolute;width:13px;height:13px;transform:translate(-50%,-50%);border:2px solid #55f3eb;
outline:1px solid #152e2c;border-radius:50%;pointer-events:none}h3{margin:.6rem 0 .3rem;font-size:1.02rem}
article p{margin:.2rem 0}.key{color:#455b52;font-size:.85rem}header p{max-width:72ch}
"""


def render(export_path, completed_audit=None):
    material = PACKET / "盲审材料"
    manifest = material / "cases.json"
    raw = read(export_path)
    key = read(PACKET / "组织者留存" / "source_key.json")
    manifest_hash = sha256(manifest.read_bytes()).hexdigest()
    if manifest_hash != raw["case_manifest_sha256"] or manifest_hash != key["case_manifest_sha"]:
        raise ValueError("Mismatched packet or review export")
    cases = read(manifest)["cases"]
    spec = {r["id"]: r for r in key["cases"]}
    answers = {r["id"]: r["response"] or "" for r in raw["cases"]}
    case_ids = [r["id"] for r in cases]
    if (len(set(case_ids)) != len(cases) or set(case_ids) != set(answers)
            or len(answers) != len(raw["cases"]) or set(case_ids) != set(spec)
            or len(spec) != len(key["cases"])):
        raise ValueError("Review IDs are incomplete or duplicated")
    if raw["answered"] != sum(bool(a) for a in answers.values()) or raw["total"] != len(cases):
        raise ValueError("Inconsistent answered count")

    cards = {g: [] for g in GROUPS}
    out = PACKET / "核对结果"
    reviewed_count = raw["answered"]
    supplemental_id = None
    if completed_audit is not None:
        audit = read(completed_audit)
        note = audit["supplement"]
        if (note["source"] != "user_chat" or audit["export_sha256"] != sha256(export_path.read_bytes()).hexdigest()
                or audit["packet_sha256"] != manifest_hash or answers[note["id"]]):
            raise ValueError("Incorrect review supplement")
        audited = {r["id"]: r["manual"] for r in audit["rows"]}
        if set(audited) != set(answers) or any(a != audited[k] for k, a in answers.items() if k != note["id"]):
            raise ValueError("Unexpected changes to earlier answers")
        supplemental_id = note["id"]
        answers[supplemental_id] = audited[supplemental_id]
        reviewed_count += 1
        out = completed_audit.parent
    out.mkdir(exist_ok=True)
    for index, case in enumerate(cases, 1):
        ident = case["id"]
        src, answer = spec[ident], answers[ident]
        expected = {0: "scratch", 1: "insect"}[src["cls"]]
        label = "原预测" if src["source"] == "unmatched OOF candidate" else "原标注"
        local = material / case["local"]
        whole = material / case["whole"]
        if not (whole.is_file() and local.is_file()):
            raise FileNotFoundError(ident)
        with Image.open(local) as crop:
            w, h = crop.size
        x, y = case["local_center"]
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"Invalid review center: {ident}")
        folder = quote(Path(os.path.relpath(material, out)).as_posix()) + "/"
        reference = f'{label}　{DECISIONS[expected]}' if answer else "待填写"
        cards[group(src["source"], expected, answer)].append(
            f'<article><a class="crop" href="{folder}{quote(case["whole"])}" title="打开原图">'
            f'<img loading="lazy" src="{folder}{quote(case["local"])}" alt="第 {index} 处果皮">'
            f'<span class="mark" style="left:{100*x/w:.3f}%;top:{100*y/h:.3f}%"></span></a>'
            f'<h3>第 {index:02d} 处</h3><p>{reference}</p>'
            f'<p><b>人工判读　{DECISIONS[answer]}</b>{"（对话补答）" if ident == supplemental_id else ""}</p>'
            f'<p class="key">{escape(ident)}</p></article>'
        )
    counts = Counter({g: len(rows) for g, rows in cards.items()})
    nav = " ".join(f'<a href="#g{i}">{g}（{counts[g]}）</a>' for i, g in enumerate(GROUPS) if counts[g])
    body = "".join(f'<section id="g{i}"><h2>{g} · {counts[g]} 处</h2><div class="grid">'
                   + "".join(cards[g]) + '</div></section>' for i, g in enumerate(GROUPS) if counts[g])
    page = (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>果皮缺陷人工核对图册</title><style>{STYLE}</style>'
            f'<header><h1>果皮缺陷人工核对图册</h1><p>已判读 {reviewed_count} / {len(cases)} 处。'
            '青色圆环标出判读中心，点击局部图可看整张原图。这批样本取自训练图。'
            '缺陷类别来自一次人工判读，疑似漏标的缺陷范围仍待人工画框。</p></header>'
            f'<nav>{nav}</nav>{body}</html>')
    destination = out / "34处人工判读对照图册.html"
    destination.write_text(page, encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--completed-audit", type=Path)
    args = parser.parse_args()
    print(render(args.export, args.completed_audit))
