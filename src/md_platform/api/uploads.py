"""Upload handling.

Clients never name a server path: they upload bytes, the server picks the
location under a per-run directory, and only the extension of the original name
is honoured (MDAnalysis dispatches its parsers on it). Uploads are streamed and
aborted as soon as they exceed the cap, so an oversized file is never fully
written to disk.
"""

import logging
from pathlib import Path
from typing import Set

from fastapi import HTTPException, UploadFile, status

logger = logging.getLogger("md_ai_analyzer")

CHUNK_BYTES = 1024 * 1024

TOPOLOGY_SUFFIXES: Set[str] = {
    ".pdb",
    ".gro",
    ".psf",
    ".prmtop",
    ".parm7",
    ".tpr",
    ".pdbqt",
    ".mol2",
    ".cif",
}
TRAJECTORY_SUFFIXES: Set[str] = {
    ".xtc",
    ".trr",
    ".dcd",
    ".nc",
    ".netcdf",
    ".xyz",
    ".pdb",
    ".gro",
    ".h5",
}


def _suffix(upload: UploadFile, allowed: Set[str], role: str) -> str:
    name = Path(upload.filename or "").name
    suffix = Path(name).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported {role} format '{suffix or name}'. "
                f"Allowed: {', '.join(sorted(allowed))}."
            ),
        )
    return suffix


async def save_upload(
    upload: UploadFile,
    destination_dir: Path,
    role: str,
    max_bytes: int,
) -> Path:
    """Stream one upload into ``destination_dir`` as ``<role><suffix>``."""
    allowed = TOPOLOGY_SUFFIXES if role == "topology" else TRAJECTORY_SUFFIXES
    suffix = _suffix(upload, allowed, role)

    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / f"{role}{suffix}"

    written = 0
    try:
        with open(target, "wb") as handle:
            while chunk := await upload.read(CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"{role} file exceeds the {max_bytes // (1024 * 1024)} MB "
                            "limit for this deployment."
                        ),
                    )
                handle.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The {role} file is empty.",
        )

    logger.info("Stored %s upload (%d bytes) at %s", role, written, target)
    return target
