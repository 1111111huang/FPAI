"""Content-hash fingerprinting, shared across the training/serving boundary
gaps found in this session's investigation (BUG-066, US#173/192/193): a
model file, a training-data summary, or a live feature row can each be
hashed the same way `app/backend/agent_config_hash.py` already hashes an
AgentConfig -- sort keys, JSON dump, SHA-256 -- so two things that should
match can be compared cheaply, without needing a central version registry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def data_fingerprint(payload: dict) -> str:
    """16-hex-char fingerprint of a JSON-serializable dict. Key order and
    nesting don't matter; two dicts with the same content always match."""
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def file_fingerprint(path: str | Path) -> str:
    """16-hex-char fingerprint of a file's raw bytes -- used to detect when
    a model artifact has been silently replaced under an unchanged filename
    (the exact gap that let a stale calibrator sit next to a swapped model
    with no way to notice short of BUG-066's own manual forensic dig)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
