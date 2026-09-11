from app.rag.parse_lrc import parse_lrc_text


def test_dedupes_consecutive_repeats():
    raw = (
        "[00:01.60]hello there\n"
        "[00:03.00]hello there\n"
        "[00:03.01]hello there\n"
        "[00:03.01]how are you\n"
        "[00:05.00]how are you\n"
    )
    lines = parse_lrc_text(raw)
    assert [l.text for l in lines] == ["hello there", "how are you"]
    assert lines[0].ts == 1.60
    assert lines[1].ts == 3.01


def test_skips_empty_lines():
    raw = "[00:01.60] \n[00:01.60]actual text\n"
    lines = parse_lrc_text(raw)
    assert len(lines) == 1
    assert lines[0].text == "actual text"


def test_ignores_header_lines():
    raw = "[re:Lavf63.6.100]\n[ve:63.6.100]\n\n[00:01.60]hello\n"
    lines = parse_lrc_text(raw)
    assert len(lines) == 1
    assert lines[0].text == "hello"


def test_strips_standalone_filler_words():
    raw = "[00:01.60]um we can start with roll call um and uh then we vote\n"
    lines = parse_lrc_text(raw)
    assert lines[0].text == "we can start with roll call and then we vote"


def test_leaves_words_containing_filler_letters_alone():
    raw = "[00:01.60]duh that umbrella is uhmazing\n"
    lines = parse_lrc_text(raw)
    assert lines[0].text == "duh that umbrella is uhmazing"


def test_drops_line_that_is_only_filler():
    raw = "[00:01.60]um\n[00:02.00]actual content\n"
    lines = parse_lrc_text(raw)
    assert [l.text for l in lines] == ["actual content"]
