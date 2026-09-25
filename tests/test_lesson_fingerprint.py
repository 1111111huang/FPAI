from __future__ import annotations

import yaml

from src.agent.lesson_fingerprint import compute_model_fingerprint


_counter = [0]


def _write_selection_yaml(tmp_path, contexts: dict) -> str:
    _counter[0] += 1
    path = tmp_path / f"model_selection_{_counter[0]}.yaml"
    path.write_text(yaml.dump({"contexts": contexts}))
    return str(path)


def test_returns_none_for_a_competition_with_no_contexts_entry(tmp_path):
    path = _write_selection_yaml(tmp_path, {"E0": {"result_3way": {"model_path": "a.joblib"}}})
    assert compute_model_fingerprint("SWE", selection_path=path) is None


def test_same_entry_hashes_identically(tmp_path):
    contexts = {"E0": {"result_3way": {"model_path": "a.joblib"}, "btts": {"model_path": "b.joblib"}}}
    path = _write_selection_yaml(tmp_path, contexts)
    assert compute_model_fingerprint("E0", selection_path=path) == compute_model_fingerprint("E0", selection_path=path)


def test_changing_any_target_field_changes_the_fingerprint(tmp_path):
    path_a = _write_selection_yaml(tmp_path, {"E0": {"result_3way": {"model_path": "a.joblib"}}})
    path_b = _write_selection_yaml(tmp_path, {"E0": {"result_3way": {"model_path": "a-retrained.joblib"}}})
    assert compute_model_fingerprint("E0", selection_path=path_a) != compute_model_fingerprint("E0", selection_path=path_b)


def test_a_different_competitions_own_change_does_not_affect_this_ones_fingerprint(tmp_path):
    path_a = _write_selection_yaml(tmp_path, {
        "E0": {"result_3way": {"model_path": "a.joblib"}},
        "SP1": {"result_3way": {"model_path": "sp1-old.joblib"}},
    })
    path_b = _write_selection_yaml(tmp_path, {
        "E0": {"result_3way": {"model_path": "a.joblib"}},
        "SP1": {"result_3way": {"model_path": "sp1-retrained.joblib"}},
    })
    assert compute_model_fingerprint("E0", selection_path=path_a) == compute_model_fingerprint("E0", selection_path=path_b)
