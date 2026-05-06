from pathlib import Path

from dataset import extract_text_from_subtitles, extract_text_from_txt, normalize_text


def test_extract_text_from_txt(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("Hello   world.\n\nThis\tis  a test.\n", encoding="utf-8")
    text = extract_text_from_txt(str(p))
    assert "Hello world." in text
    assert "This is a test." in text


def test_extract_text_from_srt(tmp_path):
    p = tmp_path / "sample.srt"
    p.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello there.\n\n"
        "2\n00:00:02,100 --> 00:00:03,500\n<font color='red'>General Kenobi!</font>\n",
        encoding="utf-8",
    )
    text = extract_text_from_subtitles(str(p))
    assert "Hello there." in text
    assert "General Kenobi!" in text
    assert "-->" not in text


def test_extract_text_from_ass(tmp_path):
    p = tmp_path / "sample.ass"
    p.write_text(
        "[Script Info]\nTitle: test\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,{\\i1}Hello\\Nworld{\\i0}\n",
        encoding="utf-8",
    )
    text = extract_text_from_subtitles(str(p))
    assert "Hello" in text
    assert "world" in text
    assert "{\\i1}" not in text


def test_normalize_text():
    raw = "A\u00a0B\r\n\r\n\r\nC"
    assert normalize_text(raw) == "A B\n\nC"
