import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import srt_resegment as S  # noqa: E402

SEG = S.Segmenter(main_dict=None)


def srt(*cues):
    out = []
    for i, (a, b, text) in enumerate(cues, 1):
        out.append(f"{i}\n00:00:{a:06.3f} --> 00:00:{b:06.3f}\n{text}\n".replace(".", ","))
    return "\n".join(out)


def run(content, **kw):
    cues = S.parse_srt(content)
    rs = S.Resegmenter(SEG, S.Params(**kw))
    cues, ch = rs.run(cues)
    cues, sp = rs.split_long(cues)
    return cues, ch, sp


def texts(cues):
    return [c.text for c in cues]


def test_broken_word_rejoined():
    cues, ch, _ = run(srt((1, 3, "我們明天要去台"), (3, 5, "北看展覽")))
    assert "台北" in "".join(texts(cues)).replace(" ", "")
    assert not any(t.endswith("台") for t in texts(cues))
    assert len(ch) == 1


def test_particle_moves_back():
    cues, _, _ = run(srt((1, 3, "他說他已經吃飽"), (3, 5, "了所以不用等他")))
    assert texts(cues) == ["他說他已經吃飽了", "所以不用等他"]


def test_natural_boundary_untouched():
    cues, ch, _ = run(srt((1, 3, "今天天氣很好"), (3, 5, "人很多")))
    assert ch == []


def test_repeated_char_not_joined():
    cues, ch, _ = run(srt((1, 3, "就他們會拿麥一起來自己講"), (3, 5, "講完之後我再接")))
    assert ch == []


def test_numbers_not_glued():
    cues, ch, _ = run(srt((1, 3, "的再現率是95"), (3, 5, "97所以chromosome")))
    assert ch == [] and texts(cues) == ["的再現率是95", "97所以chromosome"]


def test_dialogue_kept():
    cues, ch, _ = run(srt((1, 3, "- 你吃飽了嗎\n- 我吃飽"), (3, 5, "了不起的成就")))
    assert ch == []
    assert cues[0].lines == ["- 你吃飽了嗎", "- 我吃飽"]


def test_tags_preserved():
    cues, ch, _ = run(srt((1, 3, "<i>我們明天要去台</i>"), (3, 5, "<i>北看展覽</i>")))
    out = S.write_srt(cues, 0, SEG)
    assert "<i>台北看展覽</i>" in out or "<i>我們明天要去台北</i>" in out


def test_unmodified_multiline_verbatim():
    cues, _, _ = run(srt((1, 3, "第一行\n第二行"), (5, 7, "別的句子")))
    assert "第一行\n第二行" in S.write_srt(cues, 0, SEG)


def test_split_long():
    cues, _, sp = run(srt((1, 5, "以嘉惠病患 接下來我們將邀請今日特別蒞臨的貴賓")), split_chars=16)
    assert len(sp) >= 1 and all(S.vis_len(c.text) <= 16 for c in cues)
    assert texts(cues)[0] == "以嘉惠病患"
    assert cues[0].end == cues[1].start


def test_replacements(tmp_path):
    f = tmp_path / "r.txt"
    f.write_text("mosaism => mosaicism\nAI => 人工智慧\n", encoding="utf-8")
    rules = S.load_replacements(str(f))
    cues = S.parse_srt(srt((1, 3, "low grade mosaism"), (3, 5, "AIR 和 AI")))
    S.apply_replacements(cues, rules)
    assert texts(cues) == ["low grade mosaicism", "AIR 和 人工智慧"]


def test_timing_cluster_spread():
    cues = S.parse_srt(srt((1, 2, "前一句"), (50, 50.1, "歡迎今日所有的貴賓來參與"),
                           (50.1, 50.2, "由宜孕醫療"), (50.3, 53, "下一句話在這裡")))
    tp = S.TimingParams()
    assert len(S.check_timing(cues, tp)) >= 2
    S.fix_timing(cues, tp)
    assert S.check_timing(cues, tp) == []
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:]))
    assert cues[0].end == 2000 and cues[3].start == 50300


def test_suggest_terms_finds_names():
    filler = ["這是一段很普通的開場白", "大家好歡迎來到今天的研討會", "我們先看一下這張圖表",
              "接下來討論臨床上的結果", "謝謝各位的耐心聆聽", "那今天的天氣非常好"] * 6
    lines = filler + ["今天請到亞斯提的專家", "所以亞斯提在實驗中", "我們用亞斯提做研究",
                      "那個亞斯提很有效", "如果吃亞斯提就會好", "醫師說亞斯提比較強"]
    cues = S.parse_srt(srt(*[(i, i + 1, t) for i, t in enumerate(lines)]))
    terms = [w for w, *_ in S.suggest_terms(cues, SEG, min_count=3)]
    assert "亞斯提" in terms
