from app.rag.filenames import parse_filename, youtube_url


def test_parses_standard_bracket_format():
    meta = parse_filename(
        "[20251118] Finance Committee： Meeting of November 17, 2025 [g_vfAOPT3Y4].en-orig.lrc"
    )
    assert meta.upload_date == "2025-11-18"
    assert meta.title == "Finance Committee： Meeting of November 17, 2025"
    assert meta.video_id == "g_vfAOPT3Y4"


def test_handles_dates_and_slashes_embedded_in_title():
    meta = parse_filename(
        "[20250403] Committee of the Whole 4⧸2⧸2025 [KB9ilVA54tA].en-orig.lrc"
    )
    assert meta.upload_date == "2025-04-03"
    assert meta.title == "Committee of the Whole 4⧸2⧸2025"
    assert meta.video_id == "KB9ilVA54tA"


def test_upload_date_can_differ_wildly_from_date_in_title():
    # Upload date and any date embedded in the title are independent -
    # we deliberately don't try to reconcile them.
    meta = parse_filename(
        "[20260507] 2024-11-07 - Annual Budget Adoption [2N3v6eh1IQM].en-orig.lrc"
    )
    assert meta.upload_date == "2026-05-07"
    assert meta.title == "2024-11-07 - Annual Budget Adoption"


def test_handles_ampersand_in_title():
    meta = parse_filename(
        "[20190627] Steering & Rules 6⧸27⧸19 Item 5 [4gYsZ0f_mJg].en-orig.lrc"
    )
    assert meta.title == "Steering & Rules 6⧸27⧸19 Item 5"
    assert meta.video_id == "4gYsZ0f_mJg"


def test_falls_back_gracefully_on_non_standard_filename():
    meta = parse_filename("20201120_Protection__Policy_Committee_111920_3_aBAj4WQ1c.enorig.lrc")
    assert meta.upload_date is None
    assert meta.video_id is None
    assert meta.title  # still gets *something* usable, just unstructured


def test_youtube_url_without_timestamp():
    assert youtube_url("abc123") == "https://www.youtube.com/watch?v=abc123"


def test_youtube_url_with_timestamp():
    assert youtube_url("abc123", start_ts=90.7) == "https://www.youtube.com/watch?v=abc123&t=90s"
