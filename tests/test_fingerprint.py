from __future__ import annotations

from pathlib import Path

from src.utils.fingerprint import data_fingerprint, file_fingerprint


def test_same_dict_gives_same_fingerprint_regardless_of_key_order():
    a = data_fingerprint({"b": 2, "a": 1})
    b = data_fingerprint({"a": 1, "b": 2})
    assert a == b


def test_different_values_give_different_fingerprints():
    a = data_fingerprint({"a": 1})
    b = data_fingerprint({"a": 2})
    assert a != b


def test_fingerprint_is_a_short_hex_string():
    fp = data_fingerprint({"a": 1})
    assert len(fp) == 16
    int(fp, 16)  # raises ValueError if not valid hex


def test_file_fingerprint_changes_when_file_content_changes(tmp_path: Path):
    f = tmp_path / "model.joblib"
    f.write_bytes(b"version one")
    fp1 = file_fingerprint(f)
    f.write_bytes(b"version two")
    fp2 = file_fingerprint(f)
    assert fp1 != fp2


def test_file_fingerprint_is_stable_for_unchanged_file(tmp_path: Path):
    f = tmp_path / "model.joblib"
    f.write_bytes(b"same content")
    assert file_fingerprint(f) == file_fingerprint(f)
