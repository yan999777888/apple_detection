"""Anonymous offline region review using the released cases file only."""

import csv
import hashlib
import io
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

from PIL import Image, ImageTk

OPTIONS = {
    "scratch": "划伤",
    "insect": "虫伤",
    "other": "其他／普通果皮",
    "uncertain": "看不清／无法判断",
}


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=".saving-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def encoded(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


class ResponseStore:
    def __init__(self, directory, reviewer):
        self.directory = Path(directory)
        if not re.fullmatch(r"[0-9A-Za-z\u4e00-\u9fff_-]{1,24}", reviewer or ""):
            raise ValueError("匿名代号限 1 至 24 个汉字、字母、数字、短横线或下划线。")
        self.reviewer = reviewer
        manifest = (self.directory / "cases.json").read_bytes()
        self.packet = hashlib.sha256(manifest).hexdigest()
        self.cases = json.loads(manifest)["cases"]
        self.ids = {c["id"] for c in self.cases}
        assert len(self.ids) == len(self.cases) > 0
        self.file = self.directory / "本地填写" / f"review_{reviewer}.json"
        self.file.parent.mkdir(exist_ok=True)
        self.lock = self.file.with_suffix(".lock").open("a+b")
        try:
            if self.lock.tell() == 0:
                self.lock.write(b"0")
                self.lock.flush()
            self.lock.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = (
                json.loads(self.file.read_text("utf-8")) if self.file.exists() else {}
            )
            if state and (
                state["case_manifest_sha256"] != self.packet
                or state["reviewer"] != reviewer
            ):
                raise ValueError("这份进度属于另一份材料或另一位填写人。")
            self.answers = state.get("answers", {})
            if any(
                k not in self.ids or v not in OPTIONS for k, v in self.answers.items()
            ):
                raise ValueError("保存的进度含有无法识别的条目或答案。")
        except Exception:
            self.lock.close()
            raise

    def close(self):
        self.lock.close()

    def record(self, case_id, label):
        if case_id not in self.ids or label not in OPTIONS:
            raise ValueError("未知条目或选项。")
        answers = dict(self.answers)
        answers[case_id] = label
        record = dict(
            schema=1,
            case_manifest_sha256=self.packet,
            reviewer=self.reviewer,
            answers=answers,
        )
        write_file(self.file, encoded(record))
        self.answers = answers

    def export(self):
        folder = self.directory / "导出结果"
        tag = f"review_{self.reviewer}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        rows = [
            dict(
                id=c["id"],
                response=self.answers.get(c["id"], ""),
                判读=OPTIONS.get(self.answers.get(c["id"]), "未填写"),
            )
            for c in self.cases
        ]
        data = dict(
            schema=1,
            case_manifest_sha256=self.packet,
            reviewer=self.reviewer,
            answered=len(self.answers),
            total=len(rows),
            cases=rows,
        )
        write_file(folder / f"{tag}.json", encoded(data))
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=["id", "response", "判读"])
        writer.writeheader()
        writer.writerows(rows)
        write_file(folder / f"{tag}.csv", stream.getvalue().encode("utf-8-sig"))
        return folder / f"{tag}.json", folder / f"{tag}.csv"


class Reviewer:
    def __init__(self, root, store):
        self.root, self.store = root, store
        self.ordered = store.cases
        self.index = next(
            (i for i, c in enumerate(self.ordered) if c["id"] not in store.answers), 0
        )
        self.choice = tk.StringVar()
        self.progress = tk.StringVar()
        self.status = tk.StringVar()
        self.zoom = tk.BooleanVar(value=False)
        root.title("苹果果面盲审")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main = ttk.Frame(root, padding=14)
        main.grid(sticky="nsew")
        for col in range(2):
            main.columnconfigure(col, weight=1)
        ttk.Label(
            main, textvariable=self.progress, font=("Microsoft YaHei UI", 13, "bold")
        ).grid(row=0, columnspan=2, sticky="w")
        ttk.Label(
            main, text="判断定位线中心的果面。可结合整果位置与右侧局部纹理。"
        ).grid(row=1, columnspan=2, sticky="w", pady=(3, 8))
        side = max(
            220,
            min(
                390, root.winfo_screenwidth() // 2 - 45, root.winfo_screenheight() - 340
            ),
        )
        self.side = side
        for i, caption in enumerate(("整果位置", "局部原图")):
            ttk.Label(main, text=caption).grid(row=2, column=i)
        self.views = []
        for i in range(2):
            screen = tk.Canvas(
                main, width=side, height=side, bg="#272b2d", highlightthickness=0
            )
            screen.grid(row=3, column=i, padx=3, pady=2)
            self.views.append(screen)
        ttk.Checkbutton(
            main, text="放大局部显示", variable=self.zoom, command=self.show
        ).grid(row=4, column=1, pady=3)
        answers = ttk.Frame(main)
        answers.grid(row=5, columnspan=2, pady=8)
        self.radios = {}
        for i, (key, text) in enumerate(OPTIONS.items()):
            btn = ttk.Radiobutton(
                answers,
                text=f"{i+1}  {text}",
                variable=self.choice,
                value=key,
                command=lambda v=key: self.choose(v),
            )
            btn.pack(side="left", padx=8)
            self.radios[key] = btn
            root.bind(str(i + 1), lambda event, v=key: self.choose(v))
        actions = ttk.Frame(main)
        actions.grid(row=6, columnspan=2, sticky="ew", pady=4)
        ttk.Button(actions, text="上一条", command=lambda: self.move(-1)).pack(
            side="left", padx=2
        )
        ttk.Button(actions, text="下一条", command=lambda: self.move(1)).pack(
            side="left", padx=2
        )
        ttk.Button(actions, text="下一条未填", command=self.next_blank).pack(
            side="left", padx=2
        )
        ttk.Button(actions, text="导出已填结果", command=self.export).pack(side="right")
        ttk.Label(main, textvariable=self.status, wraplength=side * 2 - 25).grid(
            row=7, columnspan=2, sticky="w", pady=(6, 0)
        )
        root.bind("<Left>", lambda event: self.move(-1))
        root.bind("<Right>", lambda event: self.move(1))
        self.show()

    def show(self):
        case = self.ordered[self.index]
        self.choice.set(self.store.answers.get(case["id"], ""))
        self.progress.set(
            f"第 {self.index+1} / {len(self.ordered)} 条　　已填 {len(self.store.answers)} 条　　编号 {case['id']}"
        )
        self.photos = []
        for i, (field, cfield) in enumerate(
            (("whole", "center"), ("local", "local_center"))
        ):
            with Image.open(self.store.directory / case[field]) as im:
                desired = (
                    self.side if i == 0 or self.zoom.get() else min(256, self.side)
                )
                displayed = im.resize(
                    (desired, desired),
                    (
                        Image.Resampling.LANCZOS
                        if desired <= im.width
                        else Image.Resampling.NEAREST
                    ),
                )
                center = [int(v * desired / im.width) for v in case[cfield]]
            pic = ImageTk.PhotoImage(displayed)
            self.photos.append(pic)
            screen = self.views[i]
            screen.delete("all")
            offset = (self.side - desired) // 2
            screen.create_image(offset, offset, image=pic, anchor="nw")
            x, y = [v + offset for v in center]
            scale = desired / (1024 if i == 0 else 256)
            gap = max(7, round(21 * scale))
            span = gap + max(4, round(9 * scale))
            for a, b, c, d in (
                (x - span, y, x - gap, y),
                (x + gap, y, x + span, y),
                (x, y - span, x, y - gap),
                (x, y + gap, x, y + span),
            ):
                screen.create_line(a, b, c, d, fill="#1b2224", width=3)
                screen.create_line(a, b, c, d, fill="#66ebe7", width=1)

    def choose(self, key):
        try:
            self.store.record(self.ordered[self.index]["id"], key)
        except Exception as exc:
            messagebox.showerror("未保存", str(exc), parent=self.root)
            self.show()
            return
        self.status.set("已保存到本文件夹。可翻页或修改。")
        self.show()

    def move(self, offset):
        self.index = max(0, min(len(self.ordered) - 1, self.index + offset))
        self.show()

    def next_blank(self):
        for offset in range(1, len(self.ordered) + 1):
            pos = (self.index + offset) % len(self.ordered)
            if self.ordered[pos]["id"] not in self.store.answers:
                self.index = pos
                self.show()
                return
        self.status.set("所有条目均已填写，可导出结果。")

    def export(self):
        try:
            a, b = self.store.export()
            messagebox.showinfo(
                "已导出",
                f"已填写 {len(self.store.answers)} / {len(self.ordered)} 条。\n\n{a}\n{b}",
                parent=self.root,
            )
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.root)


def main():
    root = tk.Tk()
    root.withdraw()
    reviewer = simpledialog.askstring(
        "苹果果面盲审", "输入匿名填写代号（如 A01），下次用同一代号接续。", parent=root
    )
    if reviewer is None:
        root.destroy()
        return
    try:
        store = ResponseStore(Path(__file__).resolve().parent, reviewer.strip())
    except Exception as exc:
        messagebox.showerror(
            "无法打开进度", f"请确认没有另一窗口使用同一代号。\n{exc}", parent=root
        )
        root.destroy()
        return
    try:
        view = Reviewer(root, store)
        root.deiconify()
        root.mainloop()
    finally:
        store.close()


if __name__ == "__main__":
    main()
