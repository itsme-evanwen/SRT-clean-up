#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT 斷句修正工具（繁體中文為主，可混英文）

AI 語音辨識產出的 SRT 常在「詞的中間」切開字幕，例如：
    「我們明天要去台」 | 「北看展覽」
本工具會檢查每兩條相鄰字幕的交界，利用中文分詞的語言模型分數判斷交界是否
把一個詞切斷；若是，就把兩段文字接回來，再於附近「最適當」的位置（標點、
詞與詞之間、不讓語助詞落單）重新斷開，並依字數比例重新分配時間軸。

用法：
    python srt_resegment.py input.srt                # 輸出 input.fixed.srt
    python srt_resegment.py input.srt -o out.srt --report changes.txt
    python srt_resegment.py input.srt --dry-run      # 只列出會修改的地方
    python srt_resegment.py input.srt --dict mywords.txt   # 自訂詞（人名、專有名詞）
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import warnings
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

# --------------------------------------------------------------------------
# 字元分類
# --------------------------------------------------------------------------
CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0002ffff]")
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


# --------------------------------------------------------------------------
# SRT 讀寫
# --------------------------------------------------------------------------
TIME_RE = re.compile(
    r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})(.*)"
)


@dataclass
class Cue:
    start: int  # ms
    end: int    # ms
    text: str   # 單行文字（多行已合併）
    extra: str = ""  # 時間軸後面的附加設定（位置等）
    orig_index: list = field(default_factory=list)


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
    if (a[-1] in ",.!?;:" and re.match(r"[A-Za-z0-9]", b[0])):
        return a + " " + b
    return a + b


def parse_srt(content: str) -> list[Cue]:
    content = content.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [l for l in block.split("\n")]
        ti = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ti is None:
            continue
        m = TIME_RE.search(lines[ti])
        if not m:
            continue
        g = m.groups()
        text = ""
        for l in lines[ti + 1:]:
            text = join_text(text, l.strip())
        cues.append(Cue(_ms(*g[0:4]), _ms(*g[4:8]), text, g[8].rstrip(), [len(cues) + 1]))
    return cues


def wrap_line(text: str, max_line: int, seg: "Segmenter") -> str:
    """若一行超過 max_line，於中間附近的詞界拆成兩行。"""
    if max_line <= 0 or vis_len(text) <= max_line:
        return text
    bounds = seg.word_boundaries(text)
    mid = len(text) / 2
    best, best_cost = None, 1e9
    for c in bounds:
        if c <= 0 or c >= len(text):
            continue
        cost = abs(c - mid)
        if text[c - 1] in BREAK_AFTER or text[c - 1] == " ":
            cost -= 3
        if text[c] in NO_LEADING:
            cost += 50
        if cost < best_cost:
            best, best_cost = c, cost
    if best is None:
        return text
    return text[:best].rstrip() + "\n" + text[best:].lstrip()


def write_srt(cues: list[Cue], max_line: int, seg: "Segmenter") -> str:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(f"{i}\n{fmt_time(c.start)} --> {fmt_time(c.end)}{c.extra}\n"
                   f"{wrap_line(c.text, max_line, seg)}\n")
    return "\n".join(out)


def vis_len(text: str) -> float:
    """顯示長度：中文 1，英數半形 0.5。"""
    n = 0.0
    for ch in text:
        n += 0.5 if ord(ch) < 0x2E80 else 1.0
    return n


def timing_len(text: str) -> float:
    """估計念出這段文字所需的「字數」，用於依比例分配時間。"""
    n = 0.0
    for tok in re.findall(r"[A-Za-z]+|\d|\S", text):
        if tok[0].isalpha() and tok.isascii():
            n += max(1.0, len(tok) / 3.0)   # 英文單字約一個音節/3字母
        elif tok in BREAK_AFTER or tok in NO_TRAILING:
            n += 0.0
        else:
            n += 1.0
    return max(n, 0.001)


# --------------------------------------------------------------------------
# 分詞 + 語言模型分數
# --------------------------------------------------------------------------
class Segmenter:
    def __init__(self, user_dict: str | None = None, en_words: str | None = None):
        jieba.initialize()
        self.dt = jieba.dt
        if user_dict:
            with open(user_dict, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or parts[0].startswith("#"):
                        continue
                    w = to_simplified(parts[0])
                    freq = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 2000
                    self.dt.add_word(w, freq)
        self.logtotal = math.log(self.dt.total)
        self.en_words: set[str] | None = None
        if en_words:
            with open(en_words, encoding="utf-8") as f:
                self.en_words = {l.strip().lower() for l in f if l.strip()}

    # 每個邊 (s, e, score) 代表 text[s:e] 為一個詞
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
# 核心：交界修正
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
                orig=(0.0, 0.0)) -> float:
        p = self.p
        left, right = M[:c].rstrip(), M[c:].lstrip()
        if not left or not right:
            return 1e9
        delta = best_free - self.seg.best_score(M, E, forbid=c)
        cost = delta + p.w_dist * abs(c - b)
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

    def fix_pair(self, A: Cue, B: Cue) -> tuple[str, str] | str | None:
        """回傳 (新A文字, 新B文字)、'MERGE'，或 None（不需修改）。"""
        p = self.p
        gap = (B.start - A.end) / 1000.0
        if gap > p.max_gap or not A.text or not B.text:
            return None
        en_broken = self.seg.english_broken(A.text, B.text)
        M = A.text.rstrip() + B.text.lstrip() if en_broken else join_text(A.text, B.text)
        # 原斷點在 M 中的位置
        b = len(A.text.rstrip())
        if b < len(M) and M[b] == " ":
            b_pos_candidates = (b, b + 1)
        else:
            b_pos_candidates = (b,)
        E = self.seg.edges(M)
        best_free = self.seg.best_score(M, E)
        # 門檻：只有「原斷點切到詞中間」或「下一句以語助詞開頭」才處理
        lm_b = min(best_free - self.seg.best_score(M, E, forbid=x) for x in b_pos_candidates)
        if A.text.rstrip()[-1:] == B.text.lstrip()[:1]:
            lm_b = 0.0   # 「對|對」「講|講」這類重複字是口語，不算切斷詞
        if not (en_broken or lm_b >= p.lm_gate or self.starts_with_particle(B.text.lstrip())):
            return None
        orig = (vis_len(A.text.rstrip()), vis_len(B.text.lstrip()))
        base = min(self.cost_at(M, x, b, E, best_free, orig) for x in b_pos_candidates)
        if en_broken:
            base += 15

        best_c, best_cost = None, base
        lo, hi = max(1, b - p.max_shift), min(len(M) - 1, b + p.max_shift + 1)
        for c in range(lo, hi + 1):
            if c in b_pos_candidates:
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
        cues = [Cue(c.start, c.end, c.text, c.extra, list(c.orig_index)) for c in cues]
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
                merged = Cue(A.start, B.end, join_text(A.text, B.text), A.extra,
                             A.orig_index + B.orig_index)
                changes.append(Change(A.orig_index, B.orig_index, before, (merged.text,), "合併"))
                cues[i:i + 2] = [merged]
                # 合併後回頭再檢查與前一條的交界
                i = max(0, i - 1)
                continue
            na, nb = res
            self.retime(A, B, na, nb)
            changes.append(Change(A.orig_index, B.orig_index, before, (na, nb), "重新斷句"))
            A.text, B.text = na, nb
            i += 1
        return cues, changes

    @staticmethod
    def retime(A: Cue, B: Cue, na: str, nb: str):
        """依字數比例估計新的交界時間。"""
        if len(na) > len(A.text.rstrip()):
            # 從 B 開頭搬了文字到 A
            moved = na[len(A.text.rstrip()):]
            per = (B.end - B.start) / timing_len(B.text)
            t = B.start + timing_len(moved) * per
            t = min(t, B.end - 300)
            A.end = max(A.end, int(t))
            B.start = A.end
        else:
            moved = A.text.rstrip()[len(na):]
            per = (A.end - A.start) / timing_len(A.text)
            t = A.end - timing_len(moved) * per
            t = max(t, A.start + 300)
            B.start = min(B.start, int(t))
            A.end = B.start


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="修正 AI 產生的 SRT 字幕中把詞切斷的斷句")
    ap.add_argument("input", help="輸入 SRT 檔")
    ap.add_argument("-o", "--output", help="輸出 SRT 檔（預設 <輸入>.fixed.srt）")
    ap.add_argument("--report", help="把修改清單寫到此檔")
    ap.add_argument("--dry-run", action="store_true", help="只顯示修改，不寫檔")
    ap.add_argument("--dict", help="自訂詞典（每行：詞 [詞頻]），人名/專有名詞用")
    ap.add_argument("--en-words", help="英文單字表（每行一字），用於修正切斷的英文單字")
    ap.add_argument("--encoding", default="utf-8-sig", help="輸入檔編碼（預設 utf-8，可用 big5）")
    ap.add_argument("--max-chars", type=float, default=Params.max_chars, help="每條字幕最大長度（中文=1，英數=0.5）")
    ap.add_argument("--max-line", type=float, default=0, help="超過此長度時折成兩行（0=不折行）")
    ap.add_argument("--max-shift", type=int, default=Params.max_shift, help="新斷點離原斷點最多幾個字")
    ap.add_argument("--min-gain", type=float, default=Params.min_gain, help="修改門檻，越大越保守")
    ap.add_argument("--max-gap", type=float, default=Params.max_gap, help="間隔超過幾秒不處理")
    ap.add_argument("--no-merge", action="store_true", help="不允許把兩條字幕合併成一條")
    a = ap.parse_args(argv)

    p = Params(max_chars=a.max_chars, max_shift=a.max_shift, min_gain=a.min_gain,
               max_gap=a.max_gap, allow_merge=not a.no_merge)
    seg = Segmenter(a.dict, a.en_words)
    with open(a.input, encoding=a.encoding, errors="replace") as f:
        cues = parse_srt(f.read())
    new_cues, changes = Resegmenter(seg, p).run(cues)

    lines = []
    for ch in changes:
        ids = "#" + ",".join(map(str, ch.idx_a)) + " / #" + ",".join(map(str, ch.idx_b))
        before = "「" + "」|「".join(ch.before) + "」"
        after = "「" + "」|「".join(ch.after) + "」"
        lines.append(f"[{ch.reason}] {ids}\n    原：{before}\n    新：{after}")
    summary = f"共 {len(cues)} 條字幕，修改 {len(changes)} 處，輸出 {len(new_cues)} 條。"
    report = "\n".join(lines + [summary])
    print(report)

    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write(report + "\n")
    if not a.dry_run:
        out = a.output or re.sub(r"(\.srt)?$", ".fixed.srt", a.input, count=1, flags=re.I)
        with open(out, "w", encoding="utf-8") as f:
            f.write(write_srt(new_cues, int(a.max_line), seg))
        print(f"已輸出：{out}")


if __name__ == "__main__":
    main()
