#!/usr/bin/env python3
"""Compatibility entry point for the PokeCel viewer."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pokecel_viewer as _viewer


def _collada_up_axis(path: Path | None) -> str | None:
    """Read COLLADA's declared up-axis without depending on the importer."""
    if path is None or Path(path).suffix.lower() != ".dae":
        return None
    try:
        root = ET.parse(path).getroot()
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag[1:].split("}", 1)[0]
        tag = f"{{{ns}}}up_axis" if ns else "up_axis"
        node = root.find(f".//{tag}")
        if node is not None and node.text:
            return node.text.strip().upper()
    except Exception:
        pass
    return None


def _z_up_to_y_up() -> np.ndarray:
    # Rotate -90 degrees around X: COLLADA +Z becomes viewer +Y.
    # This also maps the usual Pokémon front direction (-Y) toward +Z,
    # i.e. toward the default camera.
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def run(path, w, h):
    """Run the viewer after normalizing COLLADA coordinate conventions."""
    original_loader = _viewer.load_meshes
    axis = _collada_up_axis(Path(path) if path is not None else None)

    def load_normalized(model_path):
        meshes = original_loader(model_path)
        if axis == "Z_UP":
            transform = _z_up_to_y_up()
            for mesh in meshes:
                mesh.apply_transform(transform)
        return meshes

    _viewer.load_meshes = load_normalized
    try:
        if axis == "Z_UP":
            print("PokeCel: normalized COLLADA Z_UP to viewer Y_UP")
        return _viewer.run(path, w, h)
    finally:
        _viewer.load_meshes = original_loader


def main():
    # Keep the existing CLI behavior while routing through the normalized run().
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Minimal cel-shading model viewer")
    ap.add_argument("model", nargs="?", type=Path, help="OBJ/GLB/GLTF/DAE/PLY/STL model")
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=800)
    a = ap.parse_args()
    if a.model and not a.model.exists():
        print("Model not found:", a.model, file=sys.stderr)
        return 2
    try:
        return run(a.model, max(a.width, 320), max(a.height, 240)) or 0
    except Exception as e:
        print("PokeCel failed:", e, file=sys.stderr)
        print("For FBX assets, use the DAE or OBJ variant instead.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
