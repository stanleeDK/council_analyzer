from app.rag.filenames import parse_filename


def test_extracts_date_and_strips_trailing_video_id():
    meta = parse_filename("20201120_Protection__Policy_Committee_111920_3_aBAj4WQ1c.enorig")
    assert meta.meeting_date == "2020-11-20"
    assert meta.meeting_title == "Protection Policy Committee 111920 3"


def test_falls_back_when_no_leading_date():
    meta = parse_filename("some_meeting_name")
    assert meta.meeting_date is None
    assert meta.meeting_title == "some meeting name"
