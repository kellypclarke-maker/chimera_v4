from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def load_env_file(
    path: Path,
    *,
    overwrite: bool = False,
    allowlist: Optional[Iterable[str]] = None,
) -> int:
    """Load KEY=VALUE lines into os.environ.

    - Ignores blank lines and comments (#...)
    - Does not print values.
    - When allowlist is provided, only keys in allowlist are loaded.

    Returns number of variables set.
    """

    p = Path(path)
    if not p.exists():
        return 0

    allowed = None
    if allowlist is not None:
        allowed = {str(k).strip() for k in allowlist if str(k).strip()}

    count = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        if not key:
            continue
        if allowed is not None and key not in allowed:
            continue
        if (not overwrite) and (key in os.environ):
            continue
        os.environ[key] = v.strip()
        count += 1
    return count

