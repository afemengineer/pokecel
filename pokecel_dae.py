#!/usr/bin/env python3
"""Compatibility helpers for rigged COLLADA assets used by PokeCel."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


def _namespace(root: ET.Element) -> str:
    if root.tag.startswith("{"):
        return root.tag[1:].split("}", 1)[0]
    return ""


def _tag(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}" if namespace else name


def staticize_skin_controllers(path: Path) -> tuple[Path, int]:
    """Replace COLLADA skin controller instances with bind-pose geometry instances.

    trimesh's COLLADA loader currently handles geometry instances but not controller
    instances. 3DS Pokémon exports are rigged and commonly instantiate their meshes
    through ``instance_controller``. For a shading experiment we only need the bind
    pose, so this rewrites a temporary DAE to instantiate each skin's source geometry
    directly while preserving its bind-shape transform and bound materials.
    """
    path = Path(path)
    if path.suffix.lower() != ".dae":
        return path, 0

    tree = ET.parse(path)
    root = tree.getroot()
    namespace = _namespace(root)
    if namespace:
        ET.register_namespace("", namespace)

    controller_tag = _tag(namespace, "controller")
    skin_tag = _tag(namespace, "skin")
    bind_shape_tag = _tag(namespace, "bind_shape_matrix")
    instance_controller_tag = _tag(namespace, "instance_controller")
    instance_geometry_tag = _tag(namespace, "instance_geometry")
    bind_material_tag = _tag(namespace, "bind_material")
    node_tag = _tag(namespace, "node")
    matrix_tag = _tag(namespace, "matrix")

    controllers: dict[str, tuple[str, str | None]] = {}
    for controller in root.iter(controller_tag):
        controller_id = controller.get("id")
        skin = controller.find(skin_tag)
        if not controller_id or skin is None:
            continue
        geometry_url = skin.get("source")
        if not geometry_url or not geometry_url.startswith("#"):
            continue
        bind_shape = skin.find(bind_shape_tag)
        matrix_text = None
        if bind_shape is not None and bind_shape.text:
            matrix_text = " ".join(bind_shape.text.split())
        controllers[f"#{controller_id}"] = (geometry_url, matrix_text)

    replaced = 0
    for parent in list(root.iter()):
        children = list(parent)
        for index, child in enumerate(children):
            if child.tag != instance_controller_tag:
                continue
            controller_url = child.get("url")
            mapping = controllers.get(controller_url or "")
            if mapping is None:
                continue

            geometry_url, matrix_text = mapping
            wrapper = ET.Element(node_tag, {"name": "PokeCel static bind pose"})
            if matrix_text:
                matrix = ET.SubElement(wrapper, matrix_tag)
                matrix.text = matrix_text

            geometry = ET.SubElement(wrapper, instance_geometry_tag, {"url": geometry_url})
            for sub in list(child):
                if sub.tag == bind_material_tag:
                    geometry.append(deepcopy(sub))

            parent.remove(child)
            parent.insert(index, wrapper)
            replaced += 1

    if replaced == 0:
        return path, 0

    output = path.with_name(f"{path.stem}__pokecel_static.dae")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output, replaced


def prepare_model(path: Path) -> Path:
    """Return a model path PokeCel can feed to trimesh."""
    path = Path(path)
    if path.suffix.lower() != ".dae":
        return path
    static_path, replaced = staticize_skin_controllers(path)
    if replaced:
        print(f"PokeCel: converted {replaced} COLLADA skin controller(s) to a static bind pose")
    return static_path
