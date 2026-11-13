"""Загрузка датасета на Windows/Linux с SHA256 из data/manifest.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.manifest.parent / manifest["file"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        with output.open("rb") as existing:
            verified = hashlib.file_digest(existing, "sha256").hexdigest() == manifest["sha256"]
        if verified:
            print(f"OK: {output} (SHA256 совпадает, загрузка не нужна)")
            return
    temporary = None
    try:
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".download", delete=False) as target:
            temporary = Path(target.name)
            with urlopen(manifest["download_mirror"], timeout=60) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(block)
                    target.write(block)
        if digest.hexdigest() != manifest["sha256"]:
            raise RuntimeError("SHA256 скачанного файла не совпадает с manifest.json")
        temporary.replace(output)
        print(f"OK: {output} ({digest.hexdigest()})")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
