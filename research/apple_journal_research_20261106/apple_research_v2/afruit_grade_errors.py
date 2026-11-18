"""Display classified severe fruit from the frozen test scores, without rerunning the model."""
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from afruit_grade_model import ROOT, file_sha
from afruit_grade_release import OUT


def build():
    frozen = json.loads((OUT / "test.json").read_text(encoding="utf-8"))
    predictions = OUT / "test_predictions.json"
    if file_sha(predictions) != frozen["predictions_sha256"]:
        raise ValueError("Predictions do not match the frozen test receipt")
    rows = json.loads(predictions.read_text(encoding="utf-8"))
    cutoff = frozen["threshold"]
    names = ["好果", "轻瑕果", "重瑕果"]
    severe = [row for row in rows if row["grade"] == 2]
    values = frozen["weighted_decision"]
    groups = []
    counts = []
    for predicted in (0, 1, 2):
        selected = sorted((r for r in severe if r["predicted"] == predicted),
                          key=lambda r: -r["probabilities"][0])
        counts.append(len(selected))
        if not selected:
            continue
        cards = []
        for row in selected:
            path = ROOT / row["path"]
            if file_sha(path) != row["sha256"]:
                raise ValueError("Source image changed")
            url = quote(Path(os.path.relpath(path, OUT)).as_posix())
            name = html.escape(path.name)
            probs = row["probabilities"]
            cards.append(f'<article><a href="{url}"><img src="{url}" alt="{name}" loading="lazy"></a>'
                         f'<strong>模型判为{names[predicted]}</strong><small>{name}</small>'
                         f'<small>好 {probs[0]:.1%} · 轻 {probs[1]:.1%} · 重 {probs[2]:.1%}</small></article>')
        groups.append(f'<section><h2>判为{names[predicted]} · {len(cards)} 张</h2><div>' + ''.join(cards) + '</div></section>')
    if sum(counts) != values["severe_n"] or counts[0] != values["severe_to_good_count"]:
        raise ValueError("Severe error count differs from test summary")
    css = ('*{box-sizing:border-box}body{margin:2rem auto;padding:0 1rem;max-width:1100px;background:#fafaf7;'
           'color:#222d26;font:16px/1.6 Inter,"Microsoft YaHei",sans-serif}div{display:grid;grid-template-columns:'
           'repeat(auto-fill,minmax(195px,1fr));gap:12px}article{padding:9px;border:1px solid #c7cec6;'
           'background:white}small,strong{display:block;overflow-wrap:anywhere}img{width:100%;height:170px;object-fit:contain}')
    page = (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>重瑕果留出测试</title><style>{css}</style><h1>重瑕果留出测试</h1><p>原数据标为重瑕的 {len(severe)} 张图片，'
            f'{counts[0]} 张被判成好果，{counts[1]} 张被判成轻瑕果，{counts[2]} 张判对。图中数值为模型的三类分数，'
            f'最终等级使用验证集冻结的规则。好果候选的重瑕分数达到 {cutoff:.2f} 时转入轻瑕。点击图片查看原图。</p>'
            + ''.join(groups) + '<p>图片来自 AFruitDB V1，Mojumdar、Mamun、Islam、Hasan，CC BY 4.0。'
            '<a href="https://data.mendeley.com/datasets/bz65dz2pbj/1">原始数据集</a></p></html>')
    (OUT / "重瑕果留出测试.html").write_text(page, encoding="utf-8")
    print(f"Verified {len(severe)} severe test images; predicted good/medium/bad: {counts}")


if __name__ == "__main__":
    build()
