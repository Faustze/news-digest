"""
Tests for poll_feedback module.
"""

from poll_feedback import _load_state, _parse_callback_data, _save_state


class TestParseCallbackData:
    def test_valid_json(self):
        data = '{"id":"abc123","r":"useful","c":"AI"}'
        result = _parse_callback_data(data)
        assert result["id"] == "abc123"
        assert result["r"] == "useful"
        assert result["c"] == "AI"

    def test_invalid_json(self):
        result = _parse_callback_data("not json")
        assert result is None

    def test_non_dict_json(self):
        result = _parse_callback_data("[1, 2, 3]")
        assert result is None

    def test_empty_string(self):
        result = _parse_callback_data("")
        assert result is None


class TestState:
    def test_load_state_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("poll_feedback.STATE_PATH", tmp_path / "state.json")
        result = _load_state()
        assert result == 0

    def test_save_and_load_state(self, tmp_path, monkeypatch):
        state_path = tmp_path / "state.json"
        monkeypatch.setattr("poll_feedback.STATE_PATH", state_path)
        _save_state(42)
        result = _load_state()
        assert result == 42

    def test_load_state_invalid(self, tmp_path, monkeypatch):
        state_path = tmp_path / "state.json"
        state_path.write_text("not json")
        monkeypatch.setattr("poll_feedback.STATE_PATH", state_path)
        result = _load_state()
        assert result == 0
