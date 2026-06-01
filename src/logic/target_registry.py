"""Central registry for forecast targets and their training contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TaskType = Literal["binary_classification", "multiclass_classification", "regression"]


@dataclass(frozen=True)
class TargetDefinition:
    """Definition of a trainable forecast target."""

    name: str
    task_type: TaskType
    label_columns: tuple[str, ...]
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    classes: tuple[str, ...] = ()
    artifact_prefix: str | None = None

    @property
    def artifact_name(self) -> str:
        """Return the artifact prefix used when saving models for this target."""
        return self.artifact_prefix or self.name


TARGET_REGISTRY: dict[str, TargetDefinition] = {
    "home_win": TargetDefinition(
        name="home_win",
        task_type="binary_classification",
        classes=("no", "yes"),
        label_columns=("fthg", "ftag"),
        primary_metric="log_loss",
        secondary_metrics=("accuracy",),
        artifact_prefix="legacy_home_win",
    ),
    "result_3way": TargetDefinition(
        name="result_3way",
        task_type="multiclass_classification",
        classes=("home", "draw", "away"),
        label_columns=("fthg", "ftag"),
        primary_metric="log_loss",
        secondary_metrics=("accuracy",),
    ),
    "btts": TargetDefinition(
        name="btts",
        task_type="binary_classification",
        classes=("no", "yes"),
        label_columns=("fthg", "ftag"),
        primary_metric="log_loss",
        secondary_metrics=("accuracy",),
    ),
    "home_goals": TargetDefinition(
        name="home_goals",
        task_type="regression",
        label_columns=("fthg",),
        primary_metric="mae",
        secondary_metrics=("rmse",),
    ),
    "away_goals": TargetDefinition(
        name="away_goals",
        task_type="regression",
        label_columns=("ftag",),
        primary_metric="mae",
        secondary_metrics=("rmse",),
    ),
    "total_goals": TargetDefinition(
        name="total_goals",
        task_type="regression",
        label_columns=("fthg", "ftag"),
        primary_metric="mae",
        secondary_metrics=("rmse",),
    ),
    "home_corners": TargetDefinition(
        name="home_corners",
        task_type="regression",
        label_columns=("hc",),
        primary_metric="mae",
        secondary_metrics=("rmse",),
    ),
    "away_corners": TargetDefinition(
        name="away_corners",
        task_type="regression",
        label_columns=("ac",),
        primary_metric="mae",
        secondary_metrics=("rmse",),
    ),
    "total_corners": TargetDefinition(
        name="total_corners",
        task_type="regression",
        label_columns=("hc", "ac"),
        primary_metric="mae",
        secondary_metrics=("rmse",),
    ),
}


def normalize_target_name(target_name: str | None) -> str:
    """Normalize user-provided target names and aliases."""
    normalized = str(target_name or "home_win").strip().lower()
    aliases = {
        "both_teams_to_score": "btts",
        "both-teams-to-score": "btts",
        "3way": "result_3way",
        "result": "result_3way",
    }
    return aliases.get(normalized, normalized)


def get_target_definition(target_name: str | None) -> TargetDefinition:
    """Return target definition or raise a helpful error."""
    normalized = normalize_target_name(target_name)
    try:
        return TARGET_REGISTRY[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(TARGET_REGISTRY))
        raise ValueError(f"Unsupported target '{target_name}'. Supported targets: {valid}") from exc


def list_target_definitions() -> list[TargetDefinition]:
    """Return registered target definitions in stable name order."""
    return [TARGET_REGISTRY[name] for name in sorted(TARGET_REGISTRY)]
