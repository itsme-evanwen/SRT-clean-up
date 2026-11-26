#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT 字幕整理工具 — 圖形介面

執行：python srt_gui.py
（Windows 可雙擊 run_gui.bat，macOS 可雙擊 run_gui.command）
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".srt_clean_up_gui.json")
SRT_TYPES = [("SRT 字幕", "*.srt"), ("所有檔案", "*.*")]
TXT_TYPES = [("文字檔", "*.txt"), ("所有檔案", "*.*")]


def open_path(path: str):
    """用系統預設程式開啟檔案或資料夾。"""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:  # pragma: no cover
        messagebox.showerror("無法開啟", f"{path}\n{e}")


class FileList(ttk.Frame):
    """可新增/移除/開啟的檔案清單（詞典、取代表）。"""

    def __init__(self, master, title: str, hint: str):
        super().__init__(master)
        ttk.Label(self, text=title).grid(row=0, column=0, sticky="w")
        ttk.Label(self, text=hint, foreground="#777").grid(row=0, column=1, columnspan=3, sticky="w", padx=6)
        self.lb = tk.Listbox(self, height=3, activestyle="none", exportselection=False)
        self.lb.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(2, 2))
        self.lb.bind("<Double-Button-1>", lambda e: self.open_selected())
        ttk.Button(self, text="新增…", width=7, command=self.add).grid(row=2, column=0, sticky="w")
        ttk.Button(self, text="移除", width=7, command=self.remove).grid(row=2, column=1, sticky="w")
        ttk.Button(self, text="開啟編輯", width=9, command=self.open_selected).grid(row=2, column=2, sticky="w")
        self.columnconfigure(3, weight=1)
        self.columnconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

    def add(self, path: str | None = None):
        paths = [path] if path else filedialog.askopenfilenames(filetypes=TXT_TYPES)
        for p in paths:
            if p and p not in self.items():
                self.lb.insert("end", p)

    def remove(self):
        for i in reversed(self.lb.curselection()):
            self.lb.delete(i)

    def open_selected(self):
        sel = self.lb.curselection()
        if sel:
            open_path(self.lb.get(sel[0]))

    def items(self) -> list[str]:
        return list(self.lb.get(0, "end"))

    def set_items(self, items):
        self.lb.delete(0, "end")
        for p in items:
            if os.path.exists(p):
                self.lb.insert("end", p)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SRT 字幕整理工具")
        self.geometry("880x760")
        self.minsize(760, 640)
        self.busy = False
        self._build()
        self._load_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names() and sys.platform.startswith("linux"):
            style.theme_use("clam")
        style.configure("Accent.TButton", font=("TkDefaultFont", 11, "bold"))

        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)

        # --- 檔案 ---
        f = ttk.LabelFrame(root, text="1. 字幕檔", padding=8)
        f.grid(row=0, column=0, sticky="ew")
        f.columnconfigure(1, weight=1)
        self.v_in = tk.StringVar()
        self.v_out = tk.StringVar()
        ttk.Label(f, text="輸入 SRT").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.v_in).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(f, text="選擇…", command=self.pick_input).grid(row=0, column=2)
        ttk.Label(f, text="輸出 SRT").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(f, textvariable=self.v_out).grid(row=1, column=1, sticky="ew", padx=6, pady=(4, 0))
        ttk.Button(f, text="另存為…", command=self.pick_output).grid(row=1, column=2, pady=(4, 0))

        # --- 詞典與取代 ---
        f = ttk.LabelFrame(root, text="2. 詞典與取代表（可省略）", padding=8)
        f.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        self.dicts = FileList(f, "自訂詞典", "人名、術語，每行「詞 詞頻」")
        self.dicts.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.repls = FileList(f, "取代表", "錯字對照，每行「錯誤 => 正確」")
        self.repls.grid(row=0, column=1, sticky="nsew")
        row = ttk.Frame(f)
        row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.v_builtin = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="使用內建台灣用詞詞典", variable=self.v_builtin).pack(side="left")
        ttk.Separator(row, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(row, text="找建議詞", command=self.find_terms).pack(side="left")
        ttk.Button(row, text="找同音詞", command=self.find_homophones).pack(side="left", padx=(6, 0))
        self.v_fuzzy = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="含近音", variable=self.v_fuzzy).pack(side="left", padx=(6, 0))

        # --- 選項 ---
        f = ttk.LabelFrame(root, text="3. 處理選項", padding=8)
        f.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.v_reseg = tk.BooleanVar(value=True)
        self.v_merge = tk.BooleanVar(value=True)
        self.v_split = tk.BooleanVar(value=False)
        self.v_split_n = tk.StringVar(value="20")
        self.v_timing = tk.BooleanVar(value=False)
        self.v_maxchars = tk.StringVar(value="18")
        self.v_wrap = tk.StringVar(value="0")
        self.v_gain = tk.StringVar(value="3.0")

        ttk.Checkbutton(f, text="修正切斷的詞（斷句修正）", variable=self.v_reseg).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(f, text="允許合併過短的字幕", variable=self.v_merge).grid(row=0, column=1, sticky="w", padx=(16, 0))
        g = ttk.Frame(f)
        g.grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(g, text="切分長字幕，超過", variable=self.v_split).pack(side="left")
        ttk.Spinbox(g, from_=10, to=60, width=4, textvariable=self.v_split_n).pack(side="left", padx=4)
        ttk.Label(g, text="字").pack(side="left")
        ttk.Checkbutton(f, text="修正時間軸（重疊、過快、過短）", variable=self.v_timing).grid(row=1, column=1, sticky="w", padx=(16, 0), pady=(4, 0))

        adv = ttk.Frame(f)
        adv.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(adv, text="每條理想長度").pack(side="left")
        ttk.Spinbox(adv, from_=8, to=40, width=4, textvariable=self.v_maxchars).pack(side="left", padx=(4, 14))
        ttk.Label(adv, text="折行長度（0=不折）").pack(side="left")
        ttk.Spinbox(adv, from_=0, to=40, width=4, textvariable=self.v_wrap).pack(side="left", padx=(4, 14))
        ttk.Label(adv, text="修改門檻（越大越保守）").pack(side="left")
        ttk.Spinbox(adv, from_=1.0, to=10.0, increment=0.5, width=5, textvariable=self.v_gain).pack(side="left", padx=4)

        # --- 執行 ---
        f = ttk.Frame(root)
        f.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.btn_preview = ttk.Button(f, text="預覽修改（不存檔）", command=lambda: self.run(dry=True))
        self.btn_preview.pack(side="left")
        self.btn_run = ttk.Button(f, text="執行並輸出", style="Accent.TButton", command=lambda: self.run(dry=False))
        self.btn_run.pack(side="left", padx=8)
        ttk.Button(f, text="開啟輸出資料夾", command=self.open_out_dir).pack(side="left")
        ttk.Button(f, text="儲存報告…", command=self.save_report).pack(side="left", padx=8)
        self.progress = ttk.Progressbar(f, mode="indeterminate", length=140)
        self.progress.pack(side="right")

        # --- 報告 ---
        f = ttk.LabelFrame(root, text="結果", padding=4)
        f.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        root.rowconfigure(4, weight=1)
        self.txt = ScrolledText(f, wrap="word", height=14, font=("TkFixedFont", 10))
        self.txt.pack(fill="both", expand=True)
        self.txt.tag_configure("head", foreground="#0b5cad", font=("TkFixedFont", 10, "bold"))
        self.txt.tag_configure("old", foreground="#a33")
        self.txt.tag_configure("new", foreground="#287a28")
        self.txt.tag_configure("err", foreground="#c00")
        self.v_status = tk.StringVar(value="選擇字幕檔後，按「預覽修改」查看會改哪些地方。")
        ttk.Label(root, textvariable=self.v_status, foreground="#555").grid(row=5, column=0, sticky="w", pady=(4, 0))

    # ------------------------------------------------------------ 檔案選擇
    def pick_input(self):
        p = filedialog.askopenfilename(filetypes=SRT_TYPES)
        if p:
            self.v_in.set(p)
            base, _ = os.path.splitext(p)
            self.v_out.set(base + ".fixed.srt")

    def pick_output(self):
        init = self.v_out.get() or self.v_in.get()
        p = filedialog.asksaveasfilename(defaultextension=".srt", filetypes=SRT_TYPES,
                                         initialdir=os.path.dirname(init) if init else None,
                                         initialfile=os.path.basename(init) if init else None)
        if p:
            self.v_out.set(p)

    def open_out_dir(self):
        p = self.v_out.get() or self.v_in.get()
        if p:
            open_path(os.path.dirname(os.path.abspath(p)))

    def save_report(self):
        text = self.txt.get("1.0", "end").strip()
        if not text:
            return
        base = os.path.splitext(self.v_in.get())[0] if self.v_in.get() else "report"
        p = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=TXT_TYPES,
                                         initialfile=os.path.basename(base) + ".changes.txt",
                                         initialdir=os.path.dirname(base) or None)
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            self.v_status.set(f"報告已儲存：{p}")

    # ------------------------------------------------------------ 參數
    def _num(self, var, name, cast=float):
        try:
            return cast(var.get())
        except ValueError:
            raise ValueError(f"「{name}」必須是數字")

    def build_args(self, dry: bool) -> list[str]:
        src = self.v_in.get().strip()
        if not src or not os.path.exists(src):
            raise ValueError("請先選擇存在的輸入 SRT 檔")
        args = [src]
        for d in self.dicts.items():
            args += ["--dict", d]
        for r in self.repls.items():
            args += ["--replace", r]
        if not self.v_builtin.get():
            args.append("--no-builtin-dict")
        if not self.v_reseg.get():
            args.append("--no-resegment")
        if not self.v_merge.get():
            args.append("--no-merge")
        if self.v_split.get():
            args += ["--split-chars", str(self._num(self.v_split_n, "切分長度"))]
        if self.v_timing.get():
            args.append("--fix-timing")
        args += ["--max-chars", str(self._num(self.v_maxchars, "每條理想長度")),
                 "--max-line", str(self._num(self.v_wrap, "折行長度")),
                 "--min-gain", str(self._num(self.v_gain, "修改門檻"))]
        if dry:
            args.append("--dry-run")
        else:
            out = self.v_out.get().strip()
            if not out:
                raise ValueError("請指定輸出檔")
            if os.path.abspath(out) == os.path.abspath(src):
                raise ValueError("輸出檔不能和輸入檔相同，以免覆蓋原始字幕")
            args += ["-o", out]
        return args

    # ------------------------------------------------------------ 執行
    def _start(self, args: list[str], label: str, on_done=None):
        if self.busy:
            return
        self.busy = True
        for b in (self.btn_preview, self.btn_run):
            b.state(["disabled"])
        self.progress.start(12)
        self.v_status.set(f"{label}中…（第一次執行要載入詞典，約需數秒）")
        self.txt.delete("1.0", "end")

        def work():
            buf = io.StringIO()
            ok = True
            try:
                import srt_resegment as S
                with redirect_stdout(buf), redirect_stderr(buf):
                    S.main(args)
            except SystemExit as e:
                ok = e.code in (None, 0)
            except Exception:
                ok = False
                buf.write("\n發生錯誤：\n" + traceback.format_exc())
            self.after(0, lambda: self._finish(buf.getvalue(), ok, label, on_done))

        threading.Thread(target=work, daemon=True).start()

    def _finish(self, text: str, ok: bool, label: str, on_done):
        self.busy = False
        self.progress.stop()
        for b in (self.btn_preview, self.btn_run):
            b.state(["!disabled"])
        self.show_report(text, ok)
        if ok:
            lines = text.splitlines()
            key = {"找建議詞": "=== 建議詞", "找同音詞": "=== 同音詞"}.get(label, "共 ")
            summary = next((l.strip("= ") for l in reversed(lines) if l.startswith(key)), "")
            self.v_status.set(f"{label}完成。{summary}")
            if on_done:
                on_done()
        else:
            self.v_status.set(f"{label}失敗，請看結果區的錯誤訊息。")

    def show_report(self, text: str, ok: bool = True):
        self.txt.delete("1.0", "end")
        for line in text.splitlines():
            tag = ()
            s = line.lstrip()
            if s.startswith("===") or s.startswith("["):
                tag = ("head",)
            elif s.startswith("原："):
                tag = ("old",)
            elif s.startswith("新：") or s.startswith("→"):
                tag = ("new",)
            elif not ok and ("錯誤" in s or "Error" in s):
                tag = ("err",)
            self.txt.insert("end", line + "\n", tag)
        self.txt.see("1.0")

    def run(self, dry: bool):
        try:
            args = self.build_args(dry)
        except ValueError as e:
            messagebox.showwarning("無法執行", str(e))
            return
        if not dry and os.path.exists(self.v_out.get()):
            if not messagebox.askyesno("覆蓋檔案", f"輸出檔已存在，要覆蓋嗎？\n{self.v_out.get()}"):
                return
        self._save_settings()

        def done():
            if not dry and messagebox.askyesno("完成", f"已輸出：\n{self.v_out.get()}\n\n要開啟輸出資料夾嗎？"):
                self.open_out_dir()

        self._start(args, "預覽" if dry else "處理", done)

    def _side_file(self, suffix: str) -> str:
        return os.path.splitext(self.v_in.get().strip())[0] + suffix

    def find_terms(self):
        try:
            args = self.build_args(dry=True)
        except ValueError as e:
            messagebox.showwarning("無法執行", str(e))
            return
        out = self._side_file(".terms.txt")
        args += ["--no-resegment", "--suggest-terms", out]

        def done():
            if messagebox.askyesno(
                    "建議詞",
                    f"已產生建議詞：\n{out}\n\n請打開檔案，刪掉不是詞的行後存檔。\n"
                    "要現在開啟，並加入「自訂詞典」清單嗎？"):
                self.dicts.add(out)
                open_path(out)

        self._start(args, "找建議詞", done)

    def find_homophones(self):
        try:
            args = self.build_args(dry=True)
        except ValueError as e:
            messagebox.showwarning("無法執行", str(e))
            return
        out = self._side_file(".homophones.txt")
        args += ["--no-resegment", "--homophones", out]
        if self.v_fuzzy.get():
            args.append("--homophones-fuzzy")

        def done():
            if messagebox.askyesno(
                    "同音詞",
                    f"已產生同音詞清單：\n{out}\n\n"
                    "每一行預設都有「# 」不會生效。確認要統一的寫法後，\n"
                    "刪掉該行的「# 」並把箭頭右邊改成正確寫法，存檔。\n\n"
                    "要現在開啟，並加入「取代表」清單嗎？"):
                self.repls.add(out)
                open_path(out)

        self._start(args, "找同音詞", done)

    # ------------------------------------------------------------ 設定
    def _settings(self) -> dict:
        return {
            "dicts": self.dicts.items(), "repls": self.repls.items(),
            "builtin": self.v_builtin.get(), "fuzzy": self.v_fuzzy.get(),
            "reseg": self.v_reseg.get(), "merge": self.v_merge.get(),
            "split": self.v_split.get(), "split_n": self.v_split_n.get(),
            "timing": self.v_timing.get(), "maxchars": self.v_maxchars.get(),
            "wrap": self.v_wrap.get(), "gain": self.v_gain.get(),
            "last_dir": os.path.dirname(self.v_in.get()),
        }

    def _save_settings(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._settings(), f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def _load_settings(self):
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                s = json.load(f)
        except (OSError, ValueError):
            return
        self.dicts.set_items(s.get("dicts", []))
        self.repls.set_items(s.get("repls", []))
        for var, key in ((self.v_builtin, "builtin"), (self.v_fuzzy, "fuzzy"), (self.v_reseg, "reseg"),
                         (self.v_merge, "merge"), (self.v_split, "split"), (self.v_timing, "timing")):
            if key in s:
                var.set(bool(s[key]))
        for var, key in ((self.v_split_n, "split_n"), (self.v_maxchars, "maxchars"),
                         (self.v_wrap, "wrap"), (self.v_gain, "gain")):
            if key in s:
                var.set(str(s[key]))

    def _on_close(self):
        self._save_settings()
        self.destroy()


def check_deps() -> bool:
    missing = []
    for mod, pkg in (("jieba", "jieba"), ("opencc", "opencc-python-reimplemented"), ("pypinyin", "pypinyin")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        r = tk.Tk()
        r.withdraw()
        messagebox.showerror("缺少套件", "請先安裝需要的套件，在命令列執行：\n\n"
                             f"pip install {' '.join(missing)}\n\n或：pip install -r requirements.txt")
        r.destroy()
        return False
    return True


def main():
    if not check_deps():
        return
    app = App()
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):   # 可把 SRT 拖到 run_gui 圖示上開啟
        app.v_in.set(sys.argv[1])
        app.v_out.set(os.path.splitext(sys.argv[1])[0] + ".fixed.srt")
    app.mainloop()


if __name__ == "__main__":
    main()
