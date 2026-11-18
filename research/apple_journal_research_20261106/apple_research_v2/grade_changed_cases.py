"""Show every automatic grade that changed on the historical 1280 regression panel."""
import argparse
import base64
from html import escape
from pathlib import Path

import cv2
import numpy as np

from afruit_grade_train import read
from afruit_visible_quality import ROOT, prepare_input


NAMES=("好果","轻瑕果","重瑕果")


def name(i):
    return "无法唯一关联" if i is None else NAMES[i]


def run(folder):
    previous={r["path"]:r for r in read(ROOT/"experiments/afruit_visible_quality_v1/test_predictions_1280.json")}
    cards=[]
    rows=sorted(read(folder/"paired_predictions.json"),key=lambda r:(r["truth"],r["path"]))
    for r in rows:
        if r["linked_grade"]==r["previous_linked"]:
            continue
        path=ROOT/r["path"]
        image=prepare_input(cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR),1280)
        earlier=previous[r["path"]]
        if r["same_regions"] and len(earlier["fruits"])==1:
            x,y,right,bottom=earlier["fruits"][0]["crop"]
            image=image[y:bottom,x:right]
        h,w=image.shape[:2]
        ratio=min(1,240/max(h,w))
        thumb=cv2.resize(image,(max(1,round(w*ratio)),max(1,round(h*ratio))),interpolation=cv2.INTER_AREA)
        ok,buf=cv2.imencode(".jpg",thumb,[cv2.IMWRITE_JPEG_QUALITY,86])
        if not ok:
            raise ValueError("Could not render a changed case")
        data=base64.b64encode(buf).decode("ascii")
        cards.append(f'<article><img alt="实际苹果裁片" src="data:image/jpeg;base64,{data}"><b>标注：{name(r["truth"])}</b>'
                     f'<div>原版 {name(r["previous_linked"])} → 本轮 {name(r["linked_grade"])}</div>'
                     f'<small>{escape(path.stem)} · {escape(r["group"])}</small></article>')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>历史留出上的改判果图</title>
<style>body{font:16px system-ui;margin:2rem auto;max-width:1100px;color:#172b21}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}article{padding:12px;border:1px solid #bacbc1}img{width:100%;height:190px;object-fit:contain;background:#eef2ef}b,small{display:block;margin-top:6px}</style>
<h1>历史留出上的改判果图</h1>'''
    html+=f'<p>共 {len(rows)} 张，展示自动裁片等级发生变化的全部 {len(cards)} 张。保留误判与纠正样图，等级真值来自 AFruitDB 的图像级标签。</p><main>'+''.join(cards)+'</main><p>原图来自 Mojumdar 等发布的 AFruitDB（CC BY 4.0），本页展示裁片并叠列项目预测。</p></html>'
    (folder/"逐果改判对照.html").write_text(html,encoding="utf8")
    print("changed",len(cards))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    run(parser.parse_args().folder)
