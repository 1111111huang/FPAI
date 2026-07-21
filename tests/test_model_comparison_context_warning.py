"""Regression tests for US#142: warn when a compare-models call spans
multiple competition contexts without an explicit --context filter.

MLflow experiment *names* aren't scoped by competition (e.g.
"FPAI_result_3way_lr_broad_v1" holds both E0's and SWE's runs for
result_3way) -- only the `tags.context` run tag distinguishes them. This
was flagged during Sweden's expansion scoping as a candidate risk, but
turned out to already be handled structurally: `get_runs_by_target(...,
context=...)` already filters correctly when a context is passed. The
residual gap is purely ergonomic -- the filter is optional and silently
blends everything when omitted, with no signal to the caller.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.model_comparison import ModelComparison


def _make_run(run_id: str, context: str, metric_value: float) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.data.tags = {"target": "result_3way", "context": context}
    run.data.metrics = {"test_log_loss": metric_value}
    run.data.params = {}
    return run


def _make_client(runs: list[MagicMock]) -> MagicMock:
    client = MagicMock()
    experiment = MagicMock()
    experiment.experiment_id = "1"
    client.search_experiments.return_value = [experiment]
    client.search_runs.return_value = runs
    return client


def test_warns_when_no_context_filter_and_runs_span_multiple_competitions(caplog) -> None:
    runs = [_make_run("r1", "E0", 1.0), _make_run("r2", "SWE", 1.1)]
    comparison = ModelComparison()
    with patch.object(comparison, "client", _make_client(runs)):
        import logging
        with caplog.at_level(logging.WARNING, logger="src.utils.model_comparison"):
            comparison.get_runs_by_target("result_3way")

    assert any("multiple competition contexts" in record.message for record in caplog.records)
    assert any("E0" in record.message and "SWE" in record.message for record in caplog.records)


def test_no_warning_when_explicit_context_passed(caplog) -> None:
    """An explicit --context already filters at the MLflow query level (via
    the tags.context filter string) -- the mock only returns E0 runs here,
    same as the real filtered query would, so there's nothing to warn about."""
    runs = [_make_run("r1", "E0", 1.0)]
    comparison = ModelComparison()
    with patch.object(comparison, "client", _make_client(runs)):
        import logging
        with caplog.at_level(logging.WARNING, logger="src.utils.model_comparison"):
            comparison.get_runs_by_target("result_3way", context="E0")

    assert not any("multiple competition contexts" in record.message for record in caplog.records)


def test_no_warning_when_no_context_but_only_one_competition_present(caplog) -> None:
    """No --context passed, but every returned run happens to share the same
    context (e.g. only E0 has ever been trained) -- nothing to warn about."""
    runs = [_make_run("r1", "E0", 1.0), _make_run("r2", "E0", 1.2)]
    comparison = ModelComparison()
    with patch.object(comparison, "client", _make_client(runs)):
        import logging
        with caplog.at_level(logging.WARNING, logger="src.utils.model_comparison"):
            comparison.get_runs_by_target("result_3way")

    assert not any("multiple competition contexts" in record.message for record in caplog.records)


def test_no_warning_when_no_runs_found(caplog) -> None:
    comparison = ModelComparison()
    with patch.object(comparison, "client", _make_client([])):
        import logging
        with caplog.at_level(logging.WARNING, logger="src.utils.model_comparison"):
            comparison.get_runs_by_target("result_3way")

    assert not any("multiple competition contexts" in record.message for record in caplog.records)
