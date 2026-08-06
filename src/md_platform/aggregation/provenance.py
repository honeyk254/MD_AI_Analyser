"""Input provenance.

Hashes are computed once per run and recorded in the RunCard: without them a
bundle cannot be tied back to the exact bytes it was produced from.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from ..schemas.run_card import FileProvenance

CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Streaming SHA256 so multi-GB trajectories are never fully buffered."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: str) -> FileProvenance:
    """Record filename, content hash and size for one input file."""
    resolved = Path(path)
    return FileProvenance(
        filename=resolved.name,
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def canonical_hash(payload: Dict[str, Any]) -> str:
    """SHA256 of a canonical JSON encoding, used for bundle hashes."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
