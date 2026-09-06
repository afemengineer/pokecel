# PokeCel

A deliberately small Python/OpenGL viewer for testing whether cel shading can recover some of the graphic character of 2D Pokemon artwork when applied to 3D character models.

This is an **experiment**, not a game engine. It avoids Blender, Godot, Unity, Assimp, and an FBX SDK.

## What it does

- Loads **OBJ, GLB, GLTF, DAE, PLY, and STL** through `trimesh`.
- Preserves separate scene mesh nodes and their transforms.
- Uses embedded/exposed diffuse textures when the importer provides them.
- Provides three live rendering modes:
  1. conventional smooth lighting (control case)
  2. simple three-band toon shading
  3. harder anime-style shading with colored shadows and restrained rim light
- Toggles an inverted-hull silhouette outline.
- Toggles smooth vs. flat face normals so the influence of normal treatment is easy to inspect.
- Saves screenshots for A/B comparisons.
- Runs a built-in demo mesh when no model is supplied, so installation can be tested immediately.
- Can resolve and download a **Pokémon X/Y** model ZIP from The Models Resource by Pokémon name or National Dex number.

## Requirements

- Python 3.10+
- A normal desktop OpenGL 3.3-capable graphics driver

No separate game engine or native asset SDK is required.

## Install

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Then install the Python packages:

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

## Fetch a Pokémon directly

The fetcher currently targets the **Pokémon X/Y** roster on The Models Resource. It discovers the current asset page rather than hard-coding one URL per Pokémon.

Download by name:

```bash
python pokecel_fetch.py pikachu
```

or by National Dex number:

```bash
python pokecel_fetch.py 25
```

Downloaded ZIPs are cached under `models/`, which is ignored by Git.

To download and immediately open the model in PokeCel:

```bash
python pokecel_fetch.py pikachu --run
```

For archives with multiple model variants, select one explicitly:

```bash
python pokecel_fetch.py pikachu --run --model PikachuM
```

Force a fresh download instead of using the cached ZIP:

```bash
python pokecel_fetch.py pikachu --refresh
```

The fetcher uses only Python's standard library for HTTP/HTML parsing, so it adds no extra pip dependencies. It may need adjustment if The Models Resource changes its page structure or blocks scripted requests.

## Run it without a model

```bash
python pokecel.py
```

You should get a colored sphere rendered with the anime shader. This is the quickest sanity check that OpenGL and the Python packages work.

## Run it with a model

```bash
python pokecel.py path/to/model.dae
```

or:

```bash
python pokecel.py path/to/model.obj
python pokecel.py path/to/model.glb
```

For VG Resource / Models Resource exports, prefer the **DAE** or **OBJ** version for this prototype.

You can also run a previously downloaded Models Resource ZIP directly:

```bash
python pokecel_zip.py path/to/archive.zip
```

### Why not FBX?

FBX support is intentionally excluded from the baseline. Reliable Python FBX import generally requires a native dependency such as Assimp or Autodesk's SDK, which defeats the goal of a small `pip install` experiment. If an asset package contains both `.fbx` and `.dae`, use the `.dae` file.

## Controls

| Input | Action |
| --- | --- |
| Left-drag | Orbit model |
| Mouse wheel | Zoom |
| `1` | Smooth lighting control |
| `2` | Basic toon shader |
| `3` | Anime-style toon shader |
| `O` | Toggle silhouette outline |
| `F` | Toggle flat / smooth normals |
| `R` | Reset view |
| `P` | Save screenshot to `screenshots/` |
| `Esc` | Quit |

## Suggested experiment

For each model, capture the same camera angle under:

1. smooth lighting + smooth normals
2. toon shader + smooth normals
3. anime shader + smooth normals
4. anime shader + flat normals
5. anime shader + outline

That separates the effect of lighting quantization, colored shadows, silhouette treatment, and normals instead of judging all changes at once.

The current anime shader is intentionally opinionated and simple. Its shadow tint, thresholds, rim light, and outline width are near the top of `pokecel.py`, so iteration is fast.

## Model/texturing limitations

This first version is intentionally small, so it does **not** yet implement:

- skeletal animation
- morph targets / facial animation
- normal maps
- per-material toon ramps
- hand-edited/custom vertex normals
- automatic FBX import
- every exotic DAE material convention

Those are good second-stage experiments after establishing whether the basic rendering change is visually meaningful.

## Legal note

Pokemon models and textures are copyrighted assets. Keep third-party game assets out of this repository. Use assets you have the right to inspect, and do not redistribute extracted Nintendo / Game Freak / Creatures content here.
