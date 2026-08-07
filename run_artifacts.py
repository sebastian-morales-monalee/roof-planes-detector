from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re


def create_run_directory(output_root: Path, experiment_name: str) -> Path:
    """Create a unique UTC-timestamped directory for one experiment run."""
    safe_name = re.sub(r"[^a-z0-9]+", "_", experiment_name.lower()).strip("_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_directory = output_root / f"{timestamp}_{safe_name}"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory
