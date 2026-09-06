#!/usr/bin/env python3
"""Run PokeCel directly from a downloaded model ZIP archive."""
from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

from pokecel import run

SUPPORTED = {".dae", ".obj", ".glb", ".gltf", ".ply", ".stl"}


def safe_extract(zf: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in zf.infolist():
        target = (destination / member.filename).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Unsafe ZIP path: {member.filename}") from exc
    zf.extractall(destination)


def candidates(root: Path) -> list[Path]:
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED),
        key=lambda p: (score(p), str(p).lower()),
    )


def score(path: Path) -> int:
    name = path.name.lower()
    suffix = path.suffix.lower()

    # Models Resource's Pokédex 3D Pro archives usually contain model.dae
    # alongside anim.dae. Prefer the actual model automatically.
    if name == "model.dae":
        return 0
    if "colladamax" in name and suffix == ".dae":
        return 1
    if suffix == ".dae" and "anim" not in name:
        return 2
    if suffix == ".obj":
        return 3
    if suffix == ".glb":
        return 4
    if suffix == ".gltf":
        return 5
    if suffix in {".ply", ".stl"}:
        return 6
    return 100


def select_model(found: list[Path], selector: str | None) -> Path:
    if not found:
        raise ValueError(
            "No supported model found in the ZIP. Expected DAE/OBJ/GLB/GLTF/PLY/STL. "
            "If this archive only contains FBX/SMD, download an archive that includes DAE or OBJ."
        )

    if selector:
        needle = selector.lower()
        matches = [p for p in found if needle in str(p).lower()]
        if not matches:
            choices = "\n".join(f"  - {p}" for p in found)
            raise ValueError(f"No model matching {selector!r}. Available models:\n{choices}")
        return matches[0]

    return found[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Open a downloaded Models Resource-style ZIP directly in PokeCel"
    )
    ap.add_argument("archive", type=Path, help="Downloaded ZIP archive")
    ap.add_argument(
        "--model",
        help="Optional substring used to choose a model when the ZIP contains several (e.g. PikachuM)",
    )
    ap.add_argument("--list", action="store_true", help="List supported models in the ZIP and exit")
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=800)
    args = ap.parse_args()

    if not args.archive.exists():
        print(f"Archive not found: {args.archive}", file=sys.stderr)
        return 2
    if not zipfile.is_zipfile(args.archive):
        print(f"Not a valid ZIP archive: {args.archive}", file=sys.stderr)
        return 2

    try:
        with tempfile.TemporaryDirectory(prefix="pokecel_") as temp:
            root = Path(temp)
            with zipfile.ZipFile(args.archive) as zf:
                safe_extract(zf, root)

            found = candidates(root)
            if args.list:
                if not found:
                    print("No supported DAE/OBJ/GLB/GLTF/PLY/STL models found.")
                    return 1
                for p in found:
                    print(p.relative_to(root))
                return 0

            model = select_model(found, args.model)
            print(f"Using model: {model.relative_to(root)}")
            return run(model, max(args.width, 320), max(args.height, 240)) or 0
    except Exception as exc:
        print(f"PokeCel ZIP loader failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
