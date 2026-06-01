"""MLflow file store cleanup for malformed experiments."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class MLflowStoreCleanup:
    """Identify and clean malformed MLflow experiments in local file store."""

    def __init__(self, mlruns_dir: str | Path = "mlruns") -> None:
        """Initialize cleanup tool.
        
        Args:
            mlruns_dir: Path to MLflow runs directory
        """
        self.mlruns_dir = Path(mlruns_dir)
        if not self.mlruns_dir.exists():
            raise FileNotFoundError(f"MLflow runs directory not found: {self.mlruns_dir}")

    def scan_experiments(self) -> dict[str, dict[str, Any]]:
        """Scan all experiment directories and identify malformed ones.
        
        Returns:
            Mapping of exp_id -> {exp_dir, has_meta, runs_count, status}
        """
        experiments = {}
        
        for exp_dir in sorted(self.mlruns_dir.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue
            
            meta_path = exp_dir / "meta.yaml"
            has_meta = meta_path.exists()
            
            # Count runs in this experiment
            runs_dir = exp_dir / "runs"
            runs_count = 0
            if runs_dir.exists():
                runs_count = len([d for d in runs_dir.iterdir() if d.is_dir()])
            
            # Try to read metadata if it exists
            exp_name = "unknown"
            if has_meta:
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = yaml.safe_load(f) or {}
                        exp_name = meta.get("name", "unknown")
                except Exception as e:
                    LOGGER.warning(f"Failed to read meta.yaml for {exp_dir.name}: {e}")
            
            status = "malformed" if not has_meta else "valid"
            
            experiments[exp_dir.name] = {
                "exp_dir": exp_dir,
                "exp_name": exp_name,
                "has_meta": has_meta,
                "runs_count": runs_count,
                "status": status,
            }
        
        return experiments

    def list_malformed(self) -> dict[str, dict[str, Any]]:
        """List all malformed experiments (missing meta.yaml).
        
        Returns:
            Mapping of exp_id -> experiment info for malformed experiments only
        """
        all_exps = self.scan_experiments()
        return {
            exp_id: info for exp_id, info in all_exps.items()
            if not info["has_meta"]
        }

    def backup_experiment(
        self,
        exp_id: str,
        backup_dir: str | Path = ".mlflow_backup",
    ) -> Path:
        """Backup an experiment directory.
        
        Args:
            exp_id: Experiment directory name (numeric ID)
            backup_dir: Directory to store backups
        
        Returns:
            Path to backup directory
        """
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        exp_dir = self.mlruns_dir / exp_id
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
        
        backup_path = backup_dir / exp_id
        if backup_path.exists():
            shutil.rmtree(backup_path)
        
        shutil.copytree(exp_dir, backup_path)
        LOGGER.info(f"Backed up experiment {exp_id} to {backup_path}")
        
        return backup_path

    def create_meta_yaml(
        self,
        exp_id: str,
        exp_name: str | None = None,
        lifecycle_stage: str = "active",
    ) -> Path:
        """Create a minimal meta.yaml for a malformed experiment.
        
        Args:
            exp_id: Experiment directory name
            exp_name: Human-readable experiment name (default: exp_id)
            lifecycle_stage: Lifecycle stage (active, deleted)
        
        Returns:
            Path to created meta.yaml
        """
        exp_dir = self.mlruns_dir / exp_id
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
        
        meta_path = exp_dir / "meta.yaml"
        if meta_path.exists():
            LOGGER.warning(f"meta.yaml already exists for {exp_id}")
            return meta_path
        
        if exp_name is None:
            exp_name = f"recovered_{exp_id}"
        
        meta_content = {
            "artifact_location": str(exp_dir),
            "experiment_id": exp_id,
            "lifecycle_stage": lifecycle_stage,
            "name": exp_name,
        }
        
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.dump(meta_content, f, default_flow_style=False)
        
        LOGGER.info(f"Created meta.yaml for experiment {exp_id} ({exp_name})")
        return meta_path

    def remove_experiment(self, exp_id: str, force: bool = False) -> None:
        """Remove an experiment directory entirely.
        
        Args:
            exp_id: Experiment directory name
            force: If False, warn on removal; if True, remove without warning
        """
        exp_dir = self.mlruns_dir / exp_id
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
        
        if not force:
            LOGGER.warning(f"Removing experiment {exp_id}: {exp_dir}")
        
        shutil.rmtree(exp_dir)
        LOGGER.info(f"Removed experiment directory {exp_id}")

    def cleanup_malformed(
        self,
        strategy: str = "recover",
        backup: bool = True,
    ) -> dict[str, str]:
        """Clean up all malformed experiments.
        
        Args:
            strategy: How to handle malformed experiments
                - "recover": Create minimal meta.yaml and keep runs
                - "remove": Delete the entire experiment directory
                - "backup_and_remove": Backup first, then remove
            backup: Whether to backup before any destructive operation
        
        Returns:
            Mapping of exp_id -> action taken
        """
        malformed = self.list_malformed()
        results = {}
        
        if not malformed:
            LOGGER.info("No malformed experiments found")
            return results
        
        LOGGER.info(f"Found {len(malformed)} malformed experiments")
        
        for exp_id, info in malformed.items():
            LOGGER.info(f"Processing {exp_id} ({info['runs_count']} runs)")
            
            if strategy == "recover":
                try:
                    self.create_meta_yaml(exp_id, exp_name=f"recovered_{exp_id}")
                    results[exp_id] = "recovered"
                except Exception as e:
                    LOGGER.error(f"Failed to recover {exp_id}: {e}")
                    results[exp_id] = f"recovery_failed: {e}"
            
            elif strategy == "remove":
                try:
                    if backup:
                        self.backup_experiment(exp_id)
                    self.remove_experiment(exp_id, force=True)
                    results[exp_id] = "removed"
                except Exception as e:
                    LOGGER.error(f"Failed to remove {exp_id}: {e}")
                    results[exp_id] = f"removal_failed: {e}"
            
            elif strategy == "backup_and_remove":
                try:
                    self.backup_experiment(exp_id)
                    self.remove_experiment(exp_id, force=True)
                    results[exp_id] = "backed_up_and_removed"
                except Exception as e:
                    LOGGER.error(f"Failed to backup/remove {exp_id}: {e}")
                    results[exp_id] = f"backup_remove_failed: {e}"
            
            else:
                results[exp_id] = f"unknown_strategy: {strategy}"
        
        return results

    def get_cleanup_summary(self) -> dict[str, Any]:
        """Generate summary of MLflow store status.
        
        Returns:
            Dictionary with store statistics
        """
        all_exps = self.scan_experiments()
        malformed = self.list_malformed()
        
        valid_exps = {k: v for k, v in all_exps.items() if v["status"] == "valid"}
        
        total_runs = sum(info["runs_count"] for info in all_exps.values())
        malformed_runs = sum(info["runs_count"] for info in malformed.values())
        
        return {
            "total_experiments": len(all_exps),
            "valid_experiments": len(valid_exps),
            "malformed_experiments": len(malformed),
            "total_runs": total_runs,
            "runs_in_malformed": malformed_runs,
            "malformed_exp_ids": list(malformed.keys()),
        }


def save_cleanup_report(
    summary: dict[str, Any],
    output_path: str | Path = "documents/mlflow_cleanup_report.txt",
) -> Path:
    """Save cleanup report to file.
    
    Args:
        summary: Dictionary from get_cleanup_summary()
        output_path: Where to save the report
    
    Returns:
        Path to saved report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "MLflow Store Cleanup Report",
        "=" * 50,
        "",
        f"Total Experiments: {summary['total_experiments']}",
        f"  Valid: {summary['valid_experiments']}",
        f"  Malformed: {summary['malformed_experiments']}",
        "",
        f"Total Runs: {summary['total_runs']}",
        f"  In Malformed: {summary['runs_in_malformed']}",
        "",
    ]
    
    if summary["malformed_exp_ids"]:
        lines.extend([
            "Malformed Experiment IDs:",
            "  " + ", ".join(summary["malformed_exp_ids"]),
        ])
    else:
        lines.append("No malformed experiments found.")
    
    report_text = "\n".join(lines)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    LOGGER.info(f"Saved cleanup report to {output_path}")
    return output_path
