#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT 字幕整理工具（繁體中文為主，可混英文）

處理 AI 語音辨識產出的 SRT：
  1. 自訂取代表：修正常見錯字、同音字、拼錯的英文術語（--replace）
  2. 斷句修正：兩條字幕交界切在詞中間（「台|北」）時接回並重新斷句
  3. 長字幕切分：超過長度的字幕在詞界切成兩條（--split-chars）
  4. 時間軸檢查：標出過快、過短、重疊的字幕；可自動重新分配（--fix-timing）
  5. 詞典輔助：內建台灣用詞詞典、建議詞（--suggest-terms）、同音詞檢查（--homophones）

保留原本的換行、對話行（- 開頭）與格式標籤（<i>、{\\an8}）。
沒被修改的字幕會原樣輸出。

用法：
    python srt_resegment.py input.srt                      # 輸出 input.fixed.srt
    python srt_resegment.py input.srt --dry-run            # 只列出修改
    python srt_resegment.py input.srt --dict mywords.txt --replace replace.txt
    python srt_resegment.py input.srt --split-chars 20 --fix-timing
    python srt_resegment.py input.srt --suggest-terms terms.txt
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=SyntaxWarning)

try:
    import jieba
except ImportError:  # pragma: no cover
    sys.exit("需要 jieba：pip install jieba")

try:
    from opencc import OpenCC
    _CC = OpenCC("t2s")
except ImportError:  # pragma: no cover
    _CC = None

jieba.setLogLevel(60)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# 字元分類
# --------------------------------------------------------------------------
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]")
ASCII_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-\.%]*")
# 句末/句中標點：斷在它「後面」很好
BREAK_AFTER = set("，。！？；：、,.!?;:…～~」』）)】》\"”’")
# 不該出現在字幕開頭的標點（應黏在前一句）
NO_LEADING = set("，。！？；：、,.!?;:…～~」』）)】》")
# 不該出現在字幕結尾的標點（應黏在後一句）
NO_TRAILING = set("「『（(【《“‘")
# 不該出現在字幕開頭的語助詞/附著詞（簡體，因為分詞用簡體）
PARTICLES = set("了吗呢吧啊呀啦喔哦嘛们着著哈耶啰咧嘞")
# 常接在數字後面的量詞/單位
MEASURE = set("年月日號号點点分秒歲岁個个元塊块萬万千百億亿次天週周倍度樓楼人位名家")
# 不適合放在字幕結尾的單字（結構助詞、介詞、連接詞，應跟後面的詞在一起）
WEAK_END = set("的之在把被将从对向于和与跟及或让给为这那每各")
# 建議詞過濾：分詞後若有這些單字獨立成詞，多半是口語片語（我想、把它、很高）而非專有名詞
STOP_CHARS = set("我你他她它們们的了在是有和跟与與把被給给讓让要會会不很也都就還还並并這这那哪個个些點点天顆颗後后前呢嗎吗吧啊麼么樣样去來来做看吃說说想到上下裡里多少高好太一再又才更最可能得地著着過过嘛呀喔")
# 格式標籤：HTML 標籤與 ASS 覆寫碼
TAG_RE = re.compile(r"<[^>\n]+>|\{[^}\n]*\}")
LEAD_TAGS_RE = re.compile(r"^(?:\s*(?:<[^>\n]+>|\{[^}\n]*\}))+\s*")
TRAIL_TAGS_RE = re.compile(r"\s*(?:(?:<[^>\n]+>|\{[^}\n]*\})\s*)+$")
DIALOGUE_RE = re.compile(r"^\s*(?:<[^>]+>|\{[^}]*\})*\s*[-–—－]")


def is_cjk(ch: str) -> bool:
    return bool(CJK_RE.match(ch))


_s_cache: dict[str, str] = {}


def to_simplified(text: str) -> str:
    """逐字轉簡體（保證長度不變，索引可以一一對應）。"""
    if _CC is None:
        return text
    out = []
    for ch in text:
        s = _s_cache.get(ch)
        if s is None:
            s = _CC.convert(ch)
            if len(s) != 1:
                s = ch
            _s_cache[ch] = s
        out.append(s)
    return "".join(out)


def vis_len(text: str) -> float:
    """顯示長度：中文 1，英數半形 0.5。"""
    return sum(0.5 if ord(ch) < 0x2E80 else 1.0 for ch in text)


def timing_len(text: str) -> float:
    """估計念出這段文字所需的「字數」（中文一字一音節，英文約 3 字母一音節）。"""
    n = 0.0
    for tok in re.findall(r"[A-Za-z]+|\d|\S", text):
        if tok[0].isalpha() and tok.isascii():
            n += max(1.0, len(tok) / 3.0)
        elif tok in BREAK_AFTER or tok in NO_TRAILING:
            n += 0.0
        else:
            n += 1.0
    return max(n, 0.001)


# --------------------------------------------------------------------------
# SRT 讀寫
# --------------------------------------------------------------------------
TIME_RE = re.compile(
    r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})(.*)"
)


@dataclass
class Cue:
    start: int                 # ms
    end: int                   # ms
    text: str                  # 去掉頭尾標籤、多行合併後的內文（用於處理）
    lines: list = field(default_factory=list)  # 原始文字行（未修改時原樣輸出）
    prefix: str = ""           # 開頭的格式標籤，如 {\an8}<i>
    suffix: str = ""           # 結尾的格式標籤，如 </i>
    locked: str = ""           # 非空 = 不參與斷句/切分（dialogue / tags）
    modified: bool = False     # 文字是否被改過
    extra: str = ""            # 時間軸後的附加設定
    orig_index: list = field(default_factory=list)

    def copy(self) -> "Cue":
        return Cue(self.start, self.end, self.text, list(self.lines), self.prefix, self.suffix,
                   self.locked, self.modified, self.extra, list(self.orig_index))


def _ms(h, m, s, f) -> int:
    f = (f + "00")[:3]
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(f)


def fmt_time(ms: int) -> str:
    ms = max(0, int(round(ms)))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def join_text(a: str, b: str) -> str:
    """接兩段文字：英數字與英數字之間補空白，其餘直接相接。"""
    a, b = a.rstrip(), b.lstrip()
    if not a:
        return b
    if not b:
        return a
    if re.match(r"[A-Za-z0-9]", a[-1]) and re.match(r"[A-Za-z0-9]", b[0]):
        return a + " " + b
    if a[-1] in ",.!?;:" and re.match(r"[A-Za-z0-9]", b[0]):
        return a + " " + b
    return a + b


def make_cue(start: int, end: int, lines: list, extra: str = "", idx=None) -> Cue:
    lines = [l.rstrip() for l in lines]
    while lines and not lines[-1].strip():
        lines.pop()
    full = "\n".join(lines)
    c = Cue(start, end, "", lines, extra=extra, orig_index=idx or [])
    # 對話格式：兩行以上、且有兩行以「-」開頭 → 不動它
    if len(lines) >= 2 and sum(1 for l in lines if DIALOGUE_RE.match(l)) >= 2:
        c.locked = "dialogue"
    m = LEAD_TAGS_RE.match(full)
    if m:
        c.prefix, full = m.group(0).strip(), full[m.end():]
    m = TRAIL_TAGS_RE.search(full)
    if m:
        c.suffix, full = m.group(0).strip(), full[:m.start()]
    if TAG_RE.search(full):
        c.locked = c.locked or "tags"   # 標籤在句子中間，移動文字可能拆壞標籤
    text = ""
    for l in full.split("\n"):
        text = join_text(text, l.strip())
    c.text = text
    return c


def read_text(path: str, encoding: str | None) -> tuple[str, str]:
    """讀檔並自動判斷編碼；回傳 (內容, 實際使用的編碼)。"""
    with open(path, "rb") as f:
        raw = f.read()
    if encoding:
        return raw.decode(encoding, errors="replace"), encoding
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16"
    if raw[:2000].count(b"\x00") > len(raw[:2000]) // 4:   # UTF-16 但沒有 BOM
        for enc in ("utf-16-le", "utf-16-be"):
            try:
                t = raw.decode(enc)
                if "-->" in t:
                    return t, enc
            except UnicodeDecodeError:
                pass
    for enc in ("utf-8-sig", "cp950", "big5hkscs", "gb18030"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    print("警告：無法判斷檔案編碼，以 UTF-8 讀取，無法解碼的字元會變成 �", file=sys.stderr)
    return raw.decode("utf-8", errors="replace"), "utf-8?"


def parse_srt(content: str) -> list[Cue]:
    """
    以「時間軸行」切分字幕，不依賴空行；缺少空行的 SRT 也能正確讀取。
    緊接在下一條時間軸行前面的純數字行視為序號。
    """
    content = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = content.split("\n")
    starts = [i for i, l in enumerate(lines) if "-->" in l and TIME_RE.search(l)]
    cues: list[Cue] = []
    for k, ti in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        body = lines[ti + 1:end]
        if k + 1 < len(starts) and body and body[-1].strip().isdigit():
            body = body[:-1]                   # 下一條的序號
        while body and not body[-1].strip():
            body.pop()
        g = TIME_RE.search(lines[ti]).groups()
        cues.append(make_cue(_ms(*g[0:4]), _ms(*g[4:8]), body, g[8].rstrip(), [len(cues) + 1]))
    return cues


def render_text(c: Cue, max_line: float, seg: "Segmenter") -> str:
    if not c.modified or c.locked:
        return "\n".join(c.lines)
    body = wrap_line(c.text, max_line, seg)
    return f"{c.prefix}{body}{c.suffix}"


def wrap_line(text: str, max_line: float, seg: "Segmenter") -> str:
    """若一行超過 max_line，於中間附近的詞界拆成兩行。"""
    if max_line <= 0 or vis_len(text) <= max_line:
        return text
    bounds = seg.word_boundaries(text)
    total = vis_len(text)
    best, best_cost = None, 1e9
    for c in bounds:
        if c <= 0 or c >= len(text):
            continue
        cost = abs(vis_len(text[:c]) - total / 2)
        if text[c - 1] in BREAK_AFTER or text[c - 1] == " ":
            cost -= 3
        if text[c] in NO_LEADING:
            cost += 50
        if cost < best_cost:
            best, best_cost = c, cost
    if best is None:
        return text
    return text[:best].rstrip() + "\n" + text[best:].lstrip()


def write_srt(cues: list[Cue], max_line: float, seg: "Segmenter") -> str:
    """輸出 SRT；內容為空的字幕（例如取代後只剩空白）會略過。"""
    out = []
    for c in cues:
        text = render_text(c, max_line, seg)
        if not TAG_RE.sub("", text).strip():
            continue
        out.append(f"{len(out) + 1}\n{fmt_time(c.start)} --> {fmt_time(c.end)}{c.extra}\n{text}\n")
    return "\n".join(out)


def count_empty(cues: list[Cue], max_line: float, seg: "Segmenter") -> int:
    return sum(1 for c in cues if not TAG_RE.sub("", render_text(c, max_line, seg)).strip())


# --------------------------------------------------------------------------
# 分詞 + 語言模型分數
# --------------------------------------------------------------------------
def find_main_dict(path: str | None) -> str | None:
    """--main-dict 指定的詞典；未指定時若程式旁有 dict.txt.big 就自動使用。"""
    if path:
        return path if path.lower() != "none" else None
    cand = os.path.join(SCRIPT_DIR, "dict.txt.big")
    return cand if os.path.exists(cand) else None


class Segmenter:
    def __init__(self, user_dicts=(), en_words: str | None = None,
                 main_dict: str | None = None, extra_words=()):
        # 每次建立獨立的分詞器，避免多次執行（如 GUI）時詞典互相累積
        self.dt = jieba.Tokenizer(main_dict) if main_dict else jieba.Tokenizer()
        self.dt.initialize()
        self.main_dict = main_dict
        for path in user_dicts or ():
            if not os.path.exists(path):
                print(f"警告：找不到詞典 {path}", file=sys.stderr)
                continue
            with open(path, encoding="utf-8-sig") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or parts[0].startswith("#"):
                        continue
                    freq = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 2000
                    self.add_word(parts[0], freq)
        for w in extra_words:
            self.add_word(w, 2000)
        self.logtotal = math.log(self.dt.total)
        self.en_words: set[str] | None = None
        if en_words:
            with open(en_words, encoding="utf-8-sig") as f:
                self.en_words = {l.strip().lower() for l in f if l.strip()}

    def add_word(self, word: str, freq: int = 2000):
        w = to_simplified(word)
        if any(is_cjk(ch) for ch in w):
            self.dt.add_word(w, freq)

    def known(self, word: str) -> bool:
        return bool(self.dt.FREQ.get(to_simplified(word)))

    # 每個邊 (e, score) 代表 text[s:e] 為一個詞
    def edges(self, text: str) -> list[list[tuple[int, float]]]:
        n = len(text)
        simp = to_simplified(text)
        E: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        i = 0
        while i < n:
            ch = text[i]
            if is_cjk(ch):
                j = i
                while j < n and is_cjk(text[j]):
                    j += 1
                run = simp[i:j]
                dag = self.dt.get_DAG(run)
                for k, ends in dag.items():
                    for x in ends:
                        w = run[k:x + 1]
                        freq = self.dt.FREQ.get(w) or 1
                        E[i + k].append((i + x + 1, math.log(freq) - self.logtotal))
                i = j
            elif re.match(r"[A-Za-z0-9]", ch):
                m = ASCII_WORD_RE.match(text, i)
                j = m.end()
                # 英文單字整個視為一個詞；也允許在單字內部斷開但代價極高
                E[i].append((j, -10.0))
                # 數字 + 量詞（2024年、3個、10點）視為一個詞
                if text[i:j].isdigit() and j < n and text[j] in MEASURE:
                    E[i].append((j + 1, -9.0))
                for k in range(i, j):
                    E[k].append((k + 1, -30.0))
                i = j
            else:
                score = 0.0 if (ch.isspace() or ch in BREAK_AFTER or ch in NO_TRAILING) else -8.0
                E[i].append((i + 1, score))
                i += 1
        return E

    def best_score(self, text: str, E=None, forbid: int | None = None) -> float:
        """最佳分詞的 log 機率；forbid=c 表示強制在位置 c 斷開。"""
        n = len(text)
        if E is None:
            E = self.edges(text)
        route = [-1e18] * (n + 1)
        route[n] = 0.0
        for i in range(n - 1, -1, -1):
            best = -1e18
            for e, sc in E[i]:
                if forbid is not None and i < forbid < e:
                    continue
                v = sc + route[e]
                if v > best:
                    best = v
            route[i] = best
        return route[0]

    def word_boundaries(self, text: str) -> list[int]:
        """最佳分詞結果的所有詞界位置。"""
        n = len(text)
        E = self.edges(text)
        route = [(-1e18, n)] * (n + 1)
        route[n] = (0.0, n)
        for i in range(n - 1, -1, -1):
            route[i] = max(((sc + route[e][0], e) for e, sc in E[i]), default=(-1e18, i + 1))
        b, i = [0], 0
        while i < n:
            i = route[i][1]
            b.append(i)
        return b

    def tokens(self, text: str) -> list[str]:
        b = self.word_boundaries(text)
        return [text[x:y] for x, y in zip(b, b[1:])]

    def english_broken(self, left: str, right: str) -> bool:
        """有英文字典時，判斷 'comp' | 'uter' 這種被切開的英文字。"""
        if not self.en_words:
            return False
        ml = re.search(r"([A-Za-z]+)$", left)
        mr = re.match(r"([A-Za-z]+)", right)
        if not (ml and mr):
            return False
        a, b = ml.group(1).lower(), mr.group(1).lower()
        return (a + b) in self.en_words and (a not in self.en_words or b not in self.en_words)


# --------------------------------------------------------------------------
# D. 自訂取代表
# --------------------------------------------------------------------------
@dataclass
class Rule:
    pattern: re.Pattern
    repl: str
    src: str
    count: int = 0


def load_replacements(path: str) -> list[Rule]:
    """
    每行一條規則：
        錯誤 => 正確          一般文字取代
        錯誤<TAB>正確         同上
        re:正規式 => 取代      正規表示式（可用 (?i) 不分大小寫、\\1 引用群組）
    以 # 開頭為註解。
    """
    rules: list[Rule] = []
    with open(path, encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "=>" in line:
                a, b = line.split("=>", 1)
            elif "\t" in line:
                a, b = line.split("\t", 1)
            else:
                print(f"警告：取代表第 {ln} 行格式錯誤（需要「錯誤 => 正確」）：{line}", file=sys.stderr)
                continue
            a, b = a.strip(), b.strip()
            if not a:
                continue
            if a.startswith("re:"):
                try:
                    pat = re.compile(a[3:])
                except re.error as e:
                    print(f"警告：取代表第 {ln} 行正規式錯誤：{e}", file=sys.stderr)
                    continue
                rules.append(Rule(pat, b, a))
            else:
                # 純英數的詞只比對完整單字，避免 AI → AIR 這種誤換
                if re.fullmatch(r"[A-Za-z0-9 _\-]+", a):
                    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])")
                else:
                    pat = re.compile(re.escape(a))
                rules.append(Rule(pat, b.replace("\\", "\\\\"), a))
    return rules


def replacement_words(rules: list[Rule]) -> list[str]:
    """取代後的正確詞（非正規式）一併加入分詞詞典。"""
    return [r.repl.replace("\\\\", "\\") for r in rules if not r.src.startswith("re:")]


def can_join(A: Cue, B: Cue) -> bool:
    """兩條字幕可以互相搬移文字：格式標籤相同，且時間順序正常。"""
    return (A.prefix, A.suffix) == (B.prefix, B.suffix) and \
        A.start <= A.end and B.start <= B.end and A.start <= B.start


def apply_replacements(cues: list[Cue], rules: list[Rule]) -> list[tuple]:
    """
    先在原始文字行上逐行取代（保留換行與標籤，字幕維持原排版），
    再檢查跨行的錯字（「移」換行「運」），有的話改在合併後的內文上取代。
    """
    log = []
    for i, c in enumerate(cues):
        before = c.text
        new_lines = []
        for l in c.lines:
            parts = re.split(f"({TAG_RE.pattern})", l)      # 標籤本身不取代
            for r in rules:
                for j in range(0, len(parts), 2):
                    parts[j], n = r.pattern.subn(r.repl, parts[j])
                    r.count += n
            new_lines.append("".join(parts))
        nc = c
        if new_lines != c.lines:
            nc = make_cue(c.start, c.end, new_lines, c.extra, c.orig_index)
            nc.modified = c.modified
            if c.modified:            # 之前已被改過的字幕，內文以處理過的為準再取代
                nc.text = c.text
        if not nc.locked:
            t = nc.text
            for r in rules:
                t, n = r.pattern.subn(r.repl, t)
                r.count += n
            if t != nc.text:
                nc.text, nc.modified = t, True
        if nc is not c or nc.text != before:
            log.append((c.orig_index, before, nc.text))
            cues[i] = nc
    return log


def apply_replacements_across(cues: list[Cue], rules: list[Rule], max_gap: float = 1.5) -> list[tuple]:
    """
    取代被切在兩條字幕之間的錯字（「移」|「運生殖中心」→「宜蘊」|「生殖中心」）。
    取代後的詞整個放到原本佔比較多的那一邊，交界時間依字數比例調整。
    """
    log = []
    for i in range(len(cues) - 1):
        A, B = cues[i], cues[i + 1]
        if A.locked or B.locked or not A.text or not B.text or not can_join(A, B):
            continue
        if (B.start - A.end) / 1000.0 > max_gap:
            continue
        a, b0 = A.text.rstrip(), B.text.lstrip()
        M = join_text(a, b0)
        bpos = len(a)
        for r in rules:
            m = next((m for m in r.pattern.finditer(M) if m.start() < bpos < m.end()), None)
            if not m:
                continue
            new = m.expand(r.repl)
            if bpos - m.start() >= m.end() - bpos:      # 詞大部分在 A
                na, nb = M[:m.start()] + new, M[m.end():].lstrip()
                moved = M[bpos:m.end()].strip()
                per = (B.end - B.start) / timing_len(b0)
                t = int(B.start + timing_len(moved) * per) if nb else B.end
                if nb:
                    A.end = max(A.end, min(t, B.end - 1))
                    B.start = A.end
            else:                                          # 詞大部分在 B
                na, nb = M[:m.start()].rstrip(), new + M[m.end():]
                moved = M[m.start():bpos].strip()
                per = (A.end - A.start) / timing_len(a)
                if na:
                    B.start = min(B.start, max(int(A.end - timing_len(moved) * per), A.start + 1))
                    A.end = B.start
            if not na or not nb:
                continue      # 整條都會被搬空的少見情況，不處理
            r.count += 1
            log.append((A.orig_index + B.orig_index, f"{a}|{b0}", f"{na}|{nb}"))
            A.text, B.text = na, nb
            A.modified = B.modified = True
            break
    return log


# --------------------------------------------------------------------------
# 斷句修正（交界）與長字幕切分
# --------------------------------------------------------------------------
@dataclass
class Params:
    max_chars: float = 18      # 一條字幕最多顯示長度（中文字=1，英數=0.5）
    min_chars: float = 2       # 低於此長度的片段會被懲罰
    max_shift: int = 8         # 新斷點最多離原斷點幾個字
    len_slack: float = 3.0     # 修正時允許字幕比原本長幾個字而不受罰
    en_phrase_pen: float = 2.0 # 在兩個英文字之間斷開的懲罰
    lm_gate: float = 3.0       # 原斷點的「切斷詞」分數需達此值才處理
    min_gain: float = 3.0      # 分數至少改善多少才修改（越大越保守）
    max_gap: float = 1.5       # 兩條字幕間隔超過此秒數視為真的停頓，不處理
    w_dist: float = 0.6        # 每移動一個字的代價
    punct_bonus: float = 3.0   # 斷在標點後的獎勵
    particle_pen: float = 6.0  # 下一句以語助詞開頭的懲罰
    weak_end_pen: float = 2.5  # 上一句以「的、在、把…」結尾的懲罰
    allow_merge: bool = True   # 找不到好斷點時，允許整句合併
    merge_cost: float = 5.0
    # 長字幕切分
    split_chars: float = 0     # 超過此長度就切分（0 = 不切）
    split_min_piece: float = 4 # 切出來的每段至少幾個字
    w_balance: float = 0.2     # 切分時兩段長度不平均的代價（每字）


@dataclass
class Change:
    idx_a: list
    idx_b: list
    before: tuple
    after: tuple
    reason: str


class Resegmenter:
    def __init__(self, seg: Segmenter, p: Params):
        self.seg = seg
        self.p = p

    def starts_with_particle(self, right: str) -> bool:
        """right 是否以落單的語助詞開頭（如「了所以」「啊 那」）。"""
        rs = to_simplified(right[:2])
        return bool(rs) and rs[0] in PARTICLES and (
            len(rs) < 2 or not is_cjk(right[1]) or not self.seg.dt.FREQ.get(rs))

    def cost_at(self, M: str, c: int, b: int, E, best_free: float,
                orig=(0.0, 0.0), w_dist: float | None = None) -> float:
        p = self.p
        left, right = M[:c].rstrip(), M[c:].lstrip()
        if not left or not right:
            return 1e9
        delta = best_free - self.seg.best_score(M, E, forbid=c)
        cost = delta + (p.w_dist if w_dist is None else w_dist) * abs(c - b)
        lc = M[c - 1]
        if lc.isspace() and c >= 2 and c < len(M) and M[c - 2].isascii() and M[c].isascii() \
                and M[c - 2].isalnum() and M[c].isalnum():
            cost += p.en_phrase_pen   # 英文詞組中間（free soil）盡量不拆
        elif lc in BREAK_AFTER or lc.isspace():
            cost -= p.punct_bonus
        if right[0] in NO_LEADING:
            cost += 20
        if left[-1] in NO_TRAILING:
            cost += 20
        ls = to_simplified(left[-2:])
        if ls and ls[-1] in WEAK_END and (len(ls) < 2 or not self.seg.dt.FREQ.get(ls)):
            cost += p.weak_end_pen
        if self.starts_with_particle(right):
            cost += p.particle_pen
        # 長度懲罰只針對「比原本更糟」的部分，原本就很長/很短的字幕不因此被改動
        for part, L0 in ((left, orig[0]), (right, orig[1])):
            L = vis_len(part)
            lim = max(p.max_chars, L0 + p.len_slack)
            if L > lim:
                cost += 2.0 * (L - lim)
            if L < p.min_chars and L < L0:
                cost += 3.0 * (p.min_chars - L)
        return cost

    # ---- 交界修正 ----
    def fix_pair(self, A: Cue, B: Cue) -> tuple[str, str] | str | None:
        """回傳 (新A文字, 新B文字)、'MERGE'，或 None（不需修改）。"""
        p = self.p
        if A.locked or B.locked or not can_join(A, B):
            return None
        gap = (B.start - A.end) / 1000.0
        if gap > p.max_gap or not A.text or not B.text:
            return None
        en_broken = self.seg.english_broken(A.text, B.text)
        M = A.text.rstrip() + B.text.lstrip() if en_broken else join_text(A.text, B.text)
        b = len(A.text.rstrip())   # 原斷點在 M 中的位置
        b_pos = (b, b + 1) if b < len(M) and M[b] == " " else (b,)
        E = self.seg.edges(M)
        best_free = self.seg.best_score(M, E)
        # 門檻：只有「原斷點切到詞中間」或「下一句以語助詞開頭」才處理
        lm_b = min(best_free - self.seg.best_score(M, E, forbid=x) for x in b_pos)
        if A.text.rstrip()[-1:] == B.text.lstrip()[:1]:
            lm_b = 0.0   # 「對|對」「講|講」這類重複字是口語，不算切斷詞
        if not (en_broken or lm_b >= p.lm_gate or self.starts_with_particle(B.text.lstrip())):
            return None
        orig = (vis_len(A.text.rstrip()), vis_len(B.text.lstrip()))
        base = min(self.cost_at(M, x, b, E, best_free, orig) for x in b_pos)
        if en_broken:
            base += 15

        best_c, best_cost = None, base
        lo, hi = max(1, b - p.max_shift), min(len(M) - 1, b + p.max_shift + 1)
        for c in range(lo, hi + 1):
            if c in b_pos:
                continue
            co = self.cost_at(M, c, b, E, best_free, orig)
            if co < best_cost:
                best_c, best_cost = c, co

        merge_ok = p.allow_merge and vis_len(M) <= p.max_chars
        # 其中一條只剩一兩個字（如單獨的「哈」「數」）時，合併的代價較低
        merge_cost = p.merge_cost if min(orig) > 2 else 1.0
        if merge_ok and merge_cost < best_cost and base - merge_cost >= p.min_gain:
            return "MERGE"
        if best_c is None or base - best_cost < p.min_gain:
            return None
        return M[:best_c].rstrip(), M[best_c:].lstrip()

    def run(self, cues: list[Cue]) -> tuple[list[Cue], list[Change]]:
        cues = [c.copy() for c in cues]
        changes: list[Change] = []
        i = 0
        while i < len(cues) - 1:
            A, B = cues[i], cues[i + 1]
            res = self.fix_pair(A, B)
            if res is None:
                i += 1
                continue
            before = (A.text, B.text)
            if res == "MERGE":
                merged = A.copy()
                merged.end, merged.text, merged.modified = B.end, join_text(A.text, B.text), True
                merged.suffix = B.suffix or A.suffix
                merged.orig_index = A.orig_index + B.orig_index
                changes.append(Change(A.orig_index, B.orig_index, before, (merged.text,), "合併"))
                cues[i:i + 2] = [merged]
                i = max(0, i - 1)   # 合併後回頭再檢查與前一條的交界
                continue
            na, nb = res
            self.retime(A, B, na, nb)
            changes.append(Change(A.orig_index, B.orig_index, before, (na, nb), "重新斷句"))
            A.text, B.text = na, nb
            A.modified = B.modified = True
            i += 1
        return cues, changes

    @staticmethod
    def retime(A: Cue, B: Cue, na: str, nb: str):
        """依字數比例估計新的交界時間。"""
        if len(na) > len(A.text.rstrip()):
            moved = na[len(A.text.rstrip()):]          # 從 B 開頭搬到 A
            per = (B.end - B.start) / timing_len(B.text)
            t = B.start + timing_len(moved) * per
            t = min(t, B.end - min(300, (B.end - B.start) // 2))
            A.end = max(A.end, int(t))
            B.start = A.end
        else:
            moved = A.text.rstrip()[len(na):]          # 從 A 結尾搬到 B
            per = (A.end - A.start) / timing_len(A.text)
            t = A.end - timing_len(moved) * per
            t = max(t, A.start + min(300, (A.end - A.start) // 2))
            B.start = min(B.start, int(t))
            A.end = B.start
        A.end = max(A.end, A.start)            # 保險：不產生負的時長
        B.start = min(max(B.start, A.end), B.end)

    # ---- C. 長字幕切分 ----
    def best_split(self, text: str) -> int | None:
        p = self.p
        E = self.seg.edges(text)
        best_free = self.seg.best_score(text, E)
        total = vis_len(text)
        # 分詞結果中「單字 | 單字」相鄰（且都不是虛詞）的位置
        bounds = self.seg.word_boundaries(text)
        single_pairs = set()
        for x, y, z in zip(bounds, bounds[1:], bounds[2:]):
            t1, t2 = text[x:y], text[y:z]
            if len(t1) == 1 == len(t2) and is_cjk(t1) and is_cjk(t2) \
                    and to_simplified(t1) not in STOP_CHARS and to_simplified(t2) not in STOP_CHARS:
                single_pairs.add(y)
        best_c, best_cost = None, 1e18
        for c in range(1, len(text)):
            left, right = text[:c].rstrip(), text[c:].lstrip()
            if vis_len(left) < p.split_min_piece or vis_len(right) < p.split_min_piece:
                continue
            if text[c - 1] == " " and c < len(text) and text[c] == " ":
                continue
            cost = self.cost_at(text, c, c, E, best_free, orig=(total, total), w_dist=0.0)
            cost += p.w_balance * abs(vis_len(left) - vis_len(right))
            if to_simplified(right[:1]) in "的地得":
                cost += 2.0
            if c in single_pairs:      # 兩個落單的字之間多半是詞典沒有的人名/術語
                cost += 2.5
            for part in (left, right):   # 切完仍太長的一段要再切，略加懲罰
                over = vis_len(part) - p.split_chars
                if over > 0:
                    cost += 0.5 * over
            if cost < best_cost:
                best_c, best_cost = c, cost
        return best_c

    def split_long(self, cues: list[Cue]) -> tuple[list[Cue], list[Change]]:
        p = self.p
        if p.split_chars <= 0:
            return cues, []
        out: list[Cue] = []
        changes: list[Change] = []
        todo = list(cues)
        todo.reverse()
        while todo:
            c = todo.pop()
            if c.locked or vis_len(c.text) <= p.split_chars:
                out.append(c)
                continue
            k = self.best_split(c.text)
            if k is None:
                out.append(c)
                continue
            lt, rt = c.text[:k].rstrip(), c.text[k:].lstrip()
            t = c.start + (c.end - c.start) * timing_len(lt) / timing_len(c.text)
            a, b = c.copy(), c.copy()
            a.end = b.start = int(round(t))
            a.text, b.text = lt, rt
            a.modified = b.modified = True
            # 頭尾標籤（{\an8}、<i>…</i>）兩段都保留，copy() 已複製
            changes.append(Change(c.orig_index, c.orig_index, (c.text,), (lt, rt), "切分長句"))
            todo.append(b)
            todo.append(a)
        return out, changes


# --------------------------------------------------------------------------
# B. 時間軸檢查與修正
# --------------------------------------------------------------------------
@dataclass
class TimingParams:
    max_cps: float = 15.0      # 每秒超過幾個字視為過快
    min_dur: float = 0.5       # 短於幾秒視為過短
    target_cps: float = 6.0    # 自動修正時的目標語速（每秒字數）
    min_gap: int = 0           # 修正時字幕之間保留的間隔（ms）


def cps(c: Cue) -> float:
    d = (c.end - c.start) / 1000.0
    return timing_len(c.text) / d if d > 0 else float("inf")


def check_timing(cues: list[Cue], tp: TimingParams) -> list[tuple]:
    """回傳 (輸出序號, 時間, 類型, 說明, 文字)。"""
    warns = []
    for i, c in enumerate(cues):
        d = (c.end - c.start) / 1000.0
        if d <= 0:
            warns.append((i + 1, c.start, "時長≤0", f"{d:.3f}s", c.text))
            continue
        if i + 1 < len(cues) and cues[i + 1].start < c.end:
            warns.append((i + 1, c.start, "重疊", f"與下一條重疊 {(c.end - cues[i+1].start)/1000:.3f}s", c.text))
        r = cps(c)
        if r > tp.max_cps:
            warns.append((i + 1, c.start, "過快", f"{r:.1f} 字/秒（{d:.2f}s）", c.text))
        elif d < tp.min_dur:
            warns.append((i + 1, c.start, "過短", f"{d:.2f}s", c.text))
    return warns


def fix_timing(cues: list[Cue], tp: TimingParams) -> list[tuple]:
    """
    修正時間軸，回傳修改紀錄：
      - 重疊：前一條的結束時間截到下一條開始
      - 過快/過短：把連續有問題的字幕視為一組，往前後的空白時間延伸，
        再依字數比例重新分配（推估值，建議人工確認）
    """
    log = []
    n = len(cues)
    for i in range(n - 1):
        a, b = cues[i], cues[i + 1]
        if b.start < a.end and a.start < b.start:
            log.append((i + 1, "重疊", f"{fmt_time(a.end)} → {fmt_time(b.start)}"))
            a.end = b.start

    def bad(c: Cue) -> bool:
        d = (c.end - c.start) / 1000.0
        return d <= 0 or cps(c) > tp.max_cps or d < tp.min_dur

    i = 0
    while i < n:
        if not bad(cues[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bad(cues[j + 1]) and cues[j + 1].start - cues[j].end < 500:
            j += 1
        group = cues[i:j + 1]
        need = sum(max(timing_len(c.text) / tp.target_cps, tp.min_dur) for c in group) * 1000
        ws, we = group[0].start, max(c.end for c in group)
        lo = cues[i - 1].end + tp.min_gap if i > 0 else 0
        hi = cues[j + 1].start - tp.min_gap if j + 1 < n else we + need
        lo = min(lo, ws)
        hi = max(hi, we)
        # 先往後延伸，不夠再往前
        if we - ws < need:
            we = min(hi, ws + need)
        if we - ws < need:
            ws = max(lo, we - need)
        if we - ws > (group[-1].end - group[0].start) + 1:
            weights = [max(timing_len(c.text), tp.min_dur * tp.target_cps) for c in group]
            tot = sum(weights)
            t = ws
            before = f"{fmt_time(group[0].start)}–{fmt_time(group[-1].end)}"
            for c, w in zip(group, weights):
                c.start = int(round(t))
                t += (we - ws) * w / tot
                c.end = int(round(t))
            after = f"{fmt_time(group[0].start)}–{fmt_time(group[-1].end)}"
            log.append((i + 1, f"重新分配 {len(group)} 條", f"{before} → {after}"))
        i = j + 1
    return log


# --------------------------------------------------------------------------
# E. 建議加入詞典的詞
# --------------------------------------------------------------------------
def ngram_stats(cues: list[Cue], max_len: int = 6):
    """整份字幕中文 n-gram 的次數與左右鄰字（字幕交界也接起來看，才抓得到被切開的詞）。"""
    runs: list[str] = []
    buf = ""
    for c in cues:
        for ch in c.text + "\n":
            if is_cjk(ch):
                buf += ch
            else:
                if len(buf) >= 2:
                    runs.append(buf)
                buf = ""
    cnt: Counter = Counter()
    left: dict = defaultdict(Counter)
    right: dict = defaultdict(Counter)
    for r in runs:
        L = len(r)
        for n in range(1, max_len + 1):
            for i in range(L - n + 1):
                w = r[i:i + n]
                cnt[w] += 1
                if n >= 2:
                    left[w][r[i - 1] if i > 0 else "^"] += 1
                    right[w][r[i + n] if i + n < L else "$"] += 1
    total = sum(v for k, v in cnt.items() if len(k) == 1) or 1
    return cnt, left, right, total


def phrase_like(w: str, seg: "Segmenter") -> bool:
    """w 是否像口語片語而非專有名詞（我想、把它、切片的時候、我們的）。"""
    toks = seg.tokens(w)
    if len(toks) > 1 and all(len(t) >= 2 for t in toks):
        return True
    return any(len(t) == 1 and t in STOP_CHARS for t in toks)


def suggest_terms(cues: list[Cue], seg: Segmenter, min_count: int = 3,
                  max_len: int = 6, top: int = 200) -> list[tuple]:
    """
    從整份字幕找出「常出現、內部結合緊密、左右搭配多樣、但詞典沒有」的詞，
    這些通常是人名、機構、藥名、術語，分詞時容易被切錯。
    回傳 [(詞, 次數, 分數, 目前分詞結果)]。
    """
    cnt, left, right, total = ngram_stats(cues, max_len)

    def entropy(c: Counter) -> float:
        s = sum(c.values())
        return -sum(v / s * math.log(v / s) for v in c.values())

    cands = []
    for w, k in cnt.items():
        if len(w) < 2 or k < min_count:
            continue
        if seg.known(w):
            continue
        # 內部凝固度：最弱的切法下，整體出現次數相對於兩半獨立出現的比值
        coh = min(k * total / (cnt[w[:i]] * cnt[w[i:]]) for i in range(1, len(w)))
        if coh < 20:
            continue
        ent = min(entropy(left[w]), entropy(right[w]))
        if ent < 0.8:
            continue
        # 可被切成多字詞的組合（我們的）、含虛詞的片語（把|它、切片|的|時候）略過
        if phrase_like(w, seg):
            continue
        toks = seg.tokens(w)
        score = k * math.log(coh) * ent
        cands.append((w, k, score, " | ".join(toks)))
    cands.sort(key=lambda x: -x[2])
    # 去掉被更長詞涵蓋、次數差不多的子字串（如「亞斯」被「亞斯提」涵蓋）
    keep: list[tuple] = []
    for w, k, s, t in cands:
        if any(w in kw and kk >= 0.8 * k for kw, kk, _, _ in keep):
            continue
        keep = [x for x in keep if not (x[0] in w and k >= 0.8 * x[1])]
        keep.append((w, k, s, t))
        if len(keep) >= top * 2:
            break
    keep.sort(key=lambda x: -x[2])
    return keep[:top]


# --------------------------------------------------------------------------
# 同音／近音詞檢查
# --------------------------------------------------------------------------
_py_cache: dict = {}


def _fuzzy(py: str) -> str:
    """常混的音：zh/z、ch/c、sh/s、n/l、-ng/-n，以及送氣/不送氣 t/d、p/b、k/g、q/j、c/z。"""
    for a, b in (("zh", "z"), ("ch", "z"), ("sh", "s"), ("t", "d"), ("p", "b"), ("k", "g"),
                 ("q", "j"), ("c", "z")):
        if py.startswith(a):
            py = b + py[len(a):]
            break
    if py.startswith("n") and not py.startswith("ng"):
        py = "l" + py[1:]
    if py.endswith("ng") and len(py) > 2:
        py = py[:-1]
    return py


def char_reading(ch: str, fuzzy: bool = False) -> str:
    """一個字最常用的讀音（不分聲調）。破音字只取常用讀音，避免「王/吳」這類誤配。"""
    key = (ch, fuzzy)
    r = _py_cache.get(key)
    if r is None:
        from pypinyin import lazy_pinyin
        rs = lazy_pinyin(ch, errors=lambda x: [x])
        r = rs[0] if rs else ch
        if fuzzy:
            r = _fuzzy(r)
        _py_cache[key] = r
    return r


def sound_key(w: str, fuzzy: bool = False) -> tuple:
    return tuple(char_reading(ch, fuzzy) for ch in w)


def find_homophones(cues: list[Cue], seg: Segmenter, fuzzy: bool = False,
                    min_count: int = 3) -> list[list[tuple[str, int]]]:
    """
    找出讀音相同（或相近）的不同寫法，例如「移運 / 宜孕 / 怡韻」「亞斯提 / 亞斯蒂」。
    以「建議詞」（常出現、詞典沒有的詞）為起點，在整份字幕中找同音的其他寫法。
    回傳每組 [(寫法, 次數), ...]，依次數排序。
    """
    try:
        import pypinyin  # noqa: F401
    except ImportError:
        sys.exit("同音詞檢查需要 pypinyin：pip install pypinyin")
    cnt, left, right, _ = ngram_stats(cues, 6)
    seeds = [w for w, *_ in suggest_terms(cues, seg, min_count=min_count, top=400)]
    lens = {len(t) for t in seeds}
    index: dict = defaultdict(set)
    for w in cnt:
        if len(w) in lens:
            index[sound_key(w, fuzzy)].add(w)

    def fragment(v: str) -> bool:
        """幾乎總是接同一個字 → 是更長詞的一部分（胚胎植 → 胚胎植入）。"""
        k = cnt[v]
        if k < 2:
            return False
        return any(c.most_common(1)[0][1] >= 0.9 * k and c.most_common(1)[0][0] not in "^$"
                   for c in (left[v], right[v]))

    def variant_ok(v: str) -> bool:
        if phrase_like(v, seg) or fragment(v):
            return False
        f = seg.dt.FREQ.get(to_simplified(v)) or 0
        return f < 50          # 本身就是常見正確詞（如「大動」）的不算

    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t in seeds:
        for v in index.get(sound_key(t, fuzzy), ()):
            if v != t and variant_ok(v):
                parent[find(v)] = find(t)
    groups: dict = defaultdict(list)
    for w in list(parent):
        groups[find(w)].append((w, cnt[w]))
    out = []
    for g in groups.values():
        g.sort(key=lambda x: -x[1])
        if len(g) >= 2 and sum(k for _, k in g) >= min_count:
            out.append(g)
    out.sort(key=lambda g: -sum(k for _, k in g))
    return out


def write_homophones(groups, path: str):
    from pypinyin import lazy_pinyin
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 同音／近音詞檢查（自動產生，請人工確認）\n"
                "# 每組是讀音相同的不同寫法，可能是 AI 辨識成不同的同音字。\n"
                "# 使用方式：把要統一的行前面的「# 」刪掉，箭頭右邊改成正確寫法，\n"
                "#           存檔後用 --replace 這個檔案重跑。預設右邊是出現最多次的寫法。\n\n")
        for g in groups:
            head = g[0][0]
            f.write(f"# {' '.join(lazy_pinyin(head))}：" + "、".join(f"{w} {k}" for w, k in g) + "\n")
            for w, k in g[1:]:
                f.write(f"# {w} => {head}\n")
            f.write("\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="整理 AI 產生的 SRT 字幕：取代錯字、修正斷句、切分長句、檢查時間軸")
    ap.add_argument("input", help="輸入 SRT 檔")
    ap.add_argument("-o", "--output", help="輸出 SRT 檔（預設 <輸入>.fixed.srt）")
    ap.add_argument("--report", help="把修改清單寫到此檔")
    ap.add_argument("--dry-run", action="store_true", help="只顯示修改，不寫檔")
    ap.add_argument("--encoding", help="輸入檔編碼（預設自動判斷 UTF-8/UTF-16/Big5）")
    g = ap.add_argument_group("詞典")
    g.add_argument("--dict", action="append", default=[], help="自訂詞典（每行：詞 [詞頻]），可指定多個")
    g.add_argument("--main-dict", help="主詞典檔（如 jieba 的 dict.txt.big）；預設自動使用程式旁的 dict.txt.big，填 none 停用")
    g.add_argument("--no-builtin-dict", action="store_true", help="不載入內建的台灣用詞詞典 tw_dict.txt")
    g.add_argument("--homophones", metavar="FILE", help="找出讀音相同的不同寫法，寫成取代表草稿")
    g.add_argument("--homophones-fuzzy", action="store_true", help="同音檢查也包含近音（zh/z、n/l、-ng/-n、t/d、p/b…）")
    g.add_argument("--en-words", help="英文單字表（每行一字），用於修正切斷的英文單字")
    g.add_argument("--suggest-terms", metavar="FILE", help="找出建議加入詞典的詞，寫到此檔")
    g.add_argument("--suggest-min", type=int, default=3, help="建議詞至少出現幾次（預設 3）")
    g = ap.add_argument_group("取代")
    g.add_argument("--replace", action="append", default=[], help="取代表（每行：錯誤 => 正確），可指定多個")
    g = ap.add_argument_group("斷句")
    g.add_argument("--max-chars", type=float, default=Params.max_chars, help="每條字幕理想最大長度（中文=1，英數=0.5）")
    g.add_argument("--max-line", type=float, default=0, help="被修改的字幕超過此長度時折成兩行（0=不折行）")
    g.add_argument("--max-shift", type=int, default=Params.max_shift, help="新斷點離原斷點最多幾個字")
    g.add_argument("--min-gain", type=float, default=Params.min_gain, help="修改門檻，越大越保守")
    g.add_argument("--max-gap", type=float, default=Params.max_gap, help="間隔超過幾秒不處理")
    g.add_argument("--no-merge", action="store_true", help="不允許把兩條字幕合併成一條")
    g.add_argument("--no-resegment", action="store_true", help="不做交界斷句修正")
    g.add_argument("--split-chars", type=float, default=0, help="超過此長度的字幕切成兩條（0=不切，建議 20~24）")
    g = ap.add_argument_group("時間軸")
    g.add_argument("--fix-timing", action="store_true", help="自動修正重疊、過快、過短的字幕時間")
    g.add_argument("--max-cps", type=float, default=TimingParams.max_cps, help="每秒超過幾個字視為過快")
    g.add_argument("--min-dur", type=float, default=TimingParams.min_dur, help="短於幾秒視為過短")
    g.add_argument("--target-cps", type=float, default=TimingParams.target_cps, help="修正時的目標語速（每秒字數）")
    g.add_argument("--timing-warnings", type=int, default=50, help="報告中最多列出幾筆時間軸警告")
    a = ap.parse_args(argv)

    p = Params(max_chars=a.max_chars, max_shift=a.max_shift, min_gain=a.min_gain,
               max_gap=a.max_gap, allow_merge=not a.no_merge, split_chars=a.split_chars)
    tp = TimingParams(max_cps=a.max_cps, min_dur=a.min_dur, target_cps=a.target_cps)

    rules: list[Rule] = []
    for path in a.replace:
        rules += load_replacements(path)
    main_dict = find_main_dict(a.main_dict)
    builtin = [] if a.no_builtin_dict else [x for x in (os.path.join(SCRIPT_DIR, "tw_dict.txt"),)
                                            if os.path.exists(x)]
    seg = Segmenter(builtin + a.dict, a.en_words, main_dict, replacement_words(rules))

    out_path = None if a.dry_run else (
        a.output or re.sub(r"(\.srt)?$", ".fixed.srt", a.input, count=1, flags=re.I))
    if out_path and os.path.exists(out_path) and os.path.samefile(out_path, a.input):
        sys.exit("錯誤：輸出檔不能和輸入檔相同（會覆蓋原始字幕），請用 -o 指定其他檔名")
    content, enc = read_text(a.input, a.encoding)
    with open(a.input, "rb") as f:
        had_bom = f.read(3) == b"\xef\xbb\xbf"
    cues = parse_srt(content)
    n_in = len(cues)
    out_lines: list[str] = [f"輸入：{a.input}（{enc}，{n_in} 條）"
                            + (f"；主詞典：{os.path.basename(main_dict)}" if main_dict else "")]
    locked = Counter(c.locked for c in cues if c.locked)
    if locked:
        out_lines.append("不處理斷句的字幕：" + "、".join(
            f"{ {'dialogue': '對話行', 'tags': '句中有標籤'}.get(k, k)} {v} 條" for k, v in locked.items()))

    # D. 取代
    rep_log = apply_replacements(cues, rules) if rules else []
    if rules:
        out_lines.append(f"\n=== 取代（{len(rep_log)} 條字幕）===")
        for r in rules:
            if r.count:
                out_lines.append(f"  {r.src} => {r.repl.replace(chr(92)*2, chr(92))}：{r.count} 次")
        for idx, b, af in rep_log:
            out_lines.append(f"  #{','.join(map(str, idx))}  「{b}」→「{af}」")

    # A + 斷句修正
    rs = Resegmenter(seg, p)
    changes: list[Change] = []
    if not a.no_resegment:
        cues, changes = rs.run(cues)
    out_lines.append(f"\n=== 斷句修正（{len(changes)} 處）===")
    for ch in changes:
        ids = "#" + ",".join(map(str, ch.idx_a)) + " / #" + ",".join(map(str, ch.idx_b))
        out_lines.append(f"[{ch.reason}] {ids}\n    原：「" + "」|「".join(ch.before)
                         + "」\n    新：「" + "」|「".join(ch.after) + "」")

    # 跨字幕的錯字（斷句修正後再檢查一次交界）
    cross_log = apply_replacements_across(cues, rules, p.max_gap) if rules else []
    if cross_log:
        out_lines.append(f"\n=== 跨字幕取代（{len(cross_log)} 處）===")
        for idx, b, af in cross_log:
            out_lines.append(f"  #{','.join(map(str, idx))}  「{b}」→「{af}」")

    # C. 長字幕切分
    cues, splits = rs.split_long(cues)
    if p.split_chars > 0:
        out_lines.append(f"\n=== 切分長句（>{p.split_chars:g} 字，{len(splits)} 處）===")
        for ch in splits:
            out_lines.append(f"[切分] #{','.join(map(str, ch.idx_a))}  「{ch.before[0]}」\n"
                             f"    →「" + "」|「".join(ch.after) + "」")

    # B. 時間軸
    warns = check_timing(cues, tp)
    kinds = Counter(w[2] for w in warns)
    out_lines.append(f"\n=== 時間軸檢查（{len(warns)} 筆："
                     + ("、".join(f"{k} {v}" for k, v in kinds.items()) or "無") + "）===")
    for w in warns[:a.timing_warnings]:
        out_lines.append(f"  #{w[0]} {fmt_time(w[1])} [{w[2]}] {w[3]}  {w[4][:30]}")
    if len(warns) > a.timing_warnings:
        out_lines.append(f"  …其餘 {len(warns) - a.timing_warnings} 筆省略（--timing-warnings 調整）")
    if a.fix_timing:
        tlog = fix_timing(cues, tp)
        out_lines.append(f"\n=== 時間軸修正（{len(tlog)} 處，時間為依字數推估）===")
        for idx, kind, detail in tlog:
            out_lines.append(f"  #{idx} [{kind}] {detail}")
        remain = check_timing(cues, tp)
        out_lines.append(f"  修正後仍有 {len(remain)} 筆警告")

    # 同音詞
    if a.homophones:
        groups = find_homophones(cues, seg, a.homophones_fuzzy, a.suggest_min)
        write_homophones(groups, a.homophones)
        out_lines.append(f"\n=== 同音詞：{len(groups)} 組，已寫入 {a.homophones} ===")
        for g in groups[:20]:
            out_lines.append("  " + " / ".join(f"{w}({k})" for w, k in g))

    # E. 建議詞
    if a.suggest_terms:
        terms = suggest_terms(cues, seg, a.suggest_min)
        with open(a.suggest_terms, "w", encoding="utf-8") as f:
            f.write("# 建議加入詞典的詞（自動產生，請人工挑選）\n"
                    "# 確認是正確的詞就保留，錯的整行刪掉，再用 --dict 這個檔案重跑。\n"
                    "# 格式：詞 詞頻   # 出現次數 | 目前被切成\n")
            for w, k, s, t in terms:
                f.write(f"{w} 2000   # {k} 次 | {t}\n")
        out_lines.append(f"\n=== 建議詞：{len(terms)} 個，已寫入 {a.suggest_terms} ===")
        out_lines.append("  " + "、".join(w for w, *_ in terms[:30]))

    n_empty = count_empty(cues, a.max_line, seg)
    if n_empty:
        out_lines.append(f"\n輸出時略過 {n_empty} 條空白字幕（取代後沒有文字）。")
    out_lines.append(f"\n共 {n_in} 條字幕；取代 {len(rep_log) + len(cross_log)} 條、斷句修正 {len(changes)} 處、"
                     f"切分 {len(splits)} 處；輸出 {len(cues) - n_empty} 條。")
    report = "\n".join(out_lines)
    print(report)

    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write(report + "\n")
    if not a.dry_run:
        out = out_path
        with open(out, "w", encoding="utf-8-sig" if had_bom else "utf-8") as f:
            f.write(write_srt(cues, a.max_line, seg))
        print(f"已輸出：{out}")


if __name__ == "__main__":
    main()
