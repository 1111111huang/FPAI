"""Central MLflow tracking-URI configuration (US#185).

MLflow's implicit default -- when nothing ever calls set_tracking_uri() at
all, which was every entry point in this codebase before this story -- is
the filesystem backend ("./mlruns"). MLflow itself deprecated that backend
(Feb 2026); in this project it also grew to 14GB/109k files, and a full
`select-best-models` scan against it took ~2 hours (documents/user_stories.md
Phase 29). configure_mlflow_tracking() switches every entry point to a
DB-backed store instead.

Deliberately NOT a historical-data migration: the old `mlruns/` file-store
is left as a frozen, read-only archive, not imported into the new store.
This is safe because `select-best-models` never re-queries mlflow for a
context's CURRENT champion metric -- that's read directly from
config/model_selection.yaml (`current_entry.get("metric_value")`) -- mlflow
is only searched for NEW candidate runs to consider promoting, and those
only ever need to be found in whichever store they were actually logged
to going forward.

Call this before any other mlflow.* call. Cheap and safe to call
repeatedly (e.g. from multiple entry points in the same process) -- no
"already configured" guard, since setting the same URI twice is a
harmless, stateless call into mlflow's own client config, not a real
reconfiguration cost.
"""

from __future__ import annotations

import mlflow

from src.utils.config_loader import load_settings


def configure_mlflow_tracking(config_path: str = "config.yaml") -> None:
    """Set MLflow's tracking URI from config.yaml's mlflow_tracking_uri."""
    settings = load_settings(config_path)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
