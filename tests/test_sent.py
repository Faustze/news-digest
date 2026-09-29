"""
Tests for news.sent (sent-news log).
"""

from datetime import date

from news.sent import drop_already_sent, load_sent, record_sent, save_sent

TODAY = date(2026, 9, 29)


def test_missing_or_corrupt_file_is_empty(tmp_path):
    assert load_sent(tmp_path / "none.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_sent(bad) == {}


def test_round_trip(tmp_path):
    path = tmp_path / "sent.json"
    sent = record_sent({}, [{"news_id": "a" * 64}], TODAY)
    save_sent(sent, path)
    assert load_sent(path) == {"a" * 16: "2026-09-29"}


def test_drop_already_sent_matches_by_id_prefix():
    sent = {"a" * 16: "2026-09-29"}
    items = [{"news_id": "a" * 64}, {"news_id": "b" * 64}]
    assert drop_already_sent(items, sent) == [{"news_id": "b" * 64}]


def test_record_prunes_old_entries():
    sent = {"old": "2026-09-20", "recent": "2026-09-27"}
    updated = record_sent(sent, [{"news_id": "c" * 64}], TODAY)
    assert "old" not in updated
    assert updated["recent"] == "2026-09-27"
    assert updated["c" * 16] == "2026-09-29"
