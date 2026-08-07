
forked from [https://github.com/picobyte/stable-diffusion-webui-wd14-tagger](https://github.com/picobyte/stable-diffusion-webui-wd14-tagger)

A standalone CLI tool that automatically tags anime-style images with
Danbooru-style tags (e.g. `1girl, solo, smile, blue hair, ...`) and writes one
`.txt` caption file next to each image — the standard caption format used for
training Stable Diffusion / anime-model LoRAs. Models are downloaded
automatically from Hugging Face on first use and cached locally.

## install

Requires Python 3.10+. By default the CPU-only build of onnxruntime is
installed; on NVIDIA machines use the `requirements-gpu.txt` variant instead
(see "Using GPU" below).

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# NVIDIA GPU / Google Colab: install this instead of requirements.txt
pip uninstall -y onnxruntime onnxruntime-gpu && pip install -r requirements-gpu.txt
```

## usage

```
usage: run.py [-h] (--dir DIR | --file FILE) [--threshold THRESHOLD]
              [--general-threshold GENERAL_THRESHOLD]
              [--character-threshold CHARACTER_THRESHOLD] [--ext EXT]
              [--overwrite] [--cpu] [--rawtag] [--recursive]
              [--exclude-tag t1,t2,t3] [--additional-tag t1,t2,t3]
              [--model MODELNAME]

options:
  -h, --help            show this help message and exit
  --dir DIR             Predictions for all images in the directory
  --file FILE           Predictions for one file
  --threshold THRESHOLD
                        Prediction threshold (default is 0.35; for PixAI:
                        general 0.3, character 0.85)
  --general-threshold GENERAL_THRESHOLD
                        PixAI only: threshold for general tags (default is
                        0.3; overrides --threshold)
  --character-threshold CHARACTER_THRESHOLD
                        PixAI only: threshold for character tags (default is
                        0.85)
  --ext EXT             Extension to add to caption file in case of dir option
                        (default is .txt)
  --overwrite           Overwrite caption file if it exists
  --cpu                 Use CPU only (default: auto-detect - use a GPU if one
                        is available, fall back to CPU otherwise)
  --rawtag              Use the raw output of the model
  --recursive           Enable recursive file search
  --exclude-tag t1,t2,t3
                        Specify tags to exclude (Need comma-separated list)
  --additional-tag t1,t2,t3
                        Specify tags to append (Need comma-separated list)
  --model MODELNAME     modelname to use for prediction (default is
                        wd14-convnextv2.v1)
```

### basic examples

Tag a single image (tags are printed to stdout):

```
python run.py --file image.jpg
```

Tag all images in a directory (writes one `.txt` caption file per image).
A progress bar shows `n/total` images tagged, the estimated time remaining,
and the overall tagging rate in images/second:

```
$ python run.py --dir path/to/images
Tagging:  55%|#####4    | 55/100 [01:12<00:59, 1.31image/s, Screenshot 21.46.08.png]
```

Notes for `--dir` mode:

- An existing caption file is **kept** by default — already-tagged images are
  skipped. Add `--overwrite` to regenerate them.
- Only `.png`, `.jpg`, `.jpeg` and `.webp` files are processed.
- Add `--recursive` to also include images in subfolders.

### adding / removing tags

Useful for LoRA datasets: inject your own trigger tag into every caption, and
strip any tag you don't want (e.g. an auto-detected character name you want to
replace with your own token).

```
# add a tag to every caption (confidence treated as 1.0)
python run.py --dir images/ --additional-tag "my_character"

# remove a tag from every caption (comma-separated, repeatable)
python run.py --dir images/ --exclude-tag "watermark,artist name,signature"
```

Both flags accept a comma-separated list and can be given multiple times.

### output format

By default tags are written in WebUI prompt style: underscores become spaces
and `(`/`)` are escaped (`1girl, long hair, artoria pendragon \(fate\)`).
Pass `--rawtag` to keep the raw Danbooru spelling instead
(`1girl, long_hair, artoria_pendragon_(fate)`).

## Supported Models

Recommended: **`pixai-tagger-v0.9`** — newest training data (Danbooru snapshot
2025-01), by far the best character recognition (~0.865 character F1), ideal
for LoRA captioning. `wd-eva02-large-tagger-v3` is a good alternative.

```
# PixAI Tagger v0.9. (released 2025, Danbooru snapshot 2025-01)
# Uses per-category thresholds: general tags 0.3, character tags 0.85.
# Character tags additionally add their series (IP) tag.
python run.py --model pixai-tagger-v0.9 --file image.jpg

# Camie Tagger / Camie Tagger v2 model. (released 2025)
python run.py --model camie-tagger --file image.jpg
python run.py --model camie-tagger-v2 --file image.jpg

# SmilingWolf large v3 model. (released 2024)
python run.py --model wd-vit-large-tagger-v3 --file image.jpg
python run.py --model wd-eva02-large-tagger-v3 --file image.jpg

# SmilingWolf v3 model. (released 2024)
python run.py --model wd-v1-4-vit-tagger.v3 --file image.jpg
python run.py --model wd-v1-4-convnext-tagger.v3 --file image.jpg
python run.py --model wd-v1-4-swinv2-tagger.v3 --file image.jpg

# SmilingWolf v2 model. (released 2023)
python run.py --model wd-v1-4-moat-tagger.v2 --file image.jpg
python run.py --model wd14-vit.v2 --file image.jpg
python run.py --model wd14-convnext.v2 --file image.jpg

# SmilingWolf v1 model. (released 2022/2023)
python run.py --model wd14-vit.v1 --file image.jpg
python run.py --model wd14-convnext.v1 --file image.jpg
python run.py --model wd14-convnextv2.v1 --file image.jpg
python run.py --model wd14-swinv2-v1 --file image.jpg

# Z3D-E621-Convnext (e621 / furry-oriented tags)
python run.py --model z3d-e621-convnext-toynya --file image.jpg
python run.py --model z3d-e621-convnext-silveroxides --file image.jpg

# ML-Danbooru (kiriyamaX, coarse generic tags only, no character tags)
python run.py --model mld-caformer.dec-5-97527 --file image.jpg
python run.py --model mld-tresnetd.6-30000 --file image.jpg
```

## PixAI tagger: per-category thresholds and series tags

The PixAI model applies its own recommended **per-category** thresholds
internally (general tags `0.3`, character tags `0.85`) instead of the single
`--threshold` used by the other models. Character tags additionally contribute
their series (IP) tag automatically — for example if `artoria pendragon
\(fate\)` is detected, `fate \(series\)` is included too.

Override them with two PixAI-only flags:

```
# looser character threshold (more character tags pass), general stays 0.3
python run.py --model pixai-tagger-v0.9 --dir images/ --character-threshold 0.5

# looser/stricter general threshold; --threshold acts as a shorthand for it
python run.py --model pixai-tagger-v0.9 --dir images/ --general-threshold 0.2
python run.py --model pixai-tagger-v0.9 --dir images/ --threshold 0.2
```

For all other models, `--threshold` (default `0.35`) keeps its original
meaning.

## example: tagging a LoRA training dataset

Typical workflow for a character LoRA (e.g. for an Anima-style anime model):

```
python run.py \
    --model pixai-tagger-v0.9 \
    --dir training_images \
    --additional-tag "my_trigger_tag" \
    --exclude-tag "watermark,signature,artist name" \
    --overwrite
```

This writes a caption `.txt` next to every image containing the auto-detected
general/character/series tags plus your trigger tag, minus noise tags.

## Using GPU (auto-detected)

GPU acceleration is **automatic**. When a model loads, the tagger asks
onnxruntime which execution providers are usable and picks the best one:

1. **Apple Silicon** - `CoreMLExecutionProvider` (Neural Engine), via the
   default `onnxruntime` package. No setup needed.
2. **NVIDIA GPU** - `CUDAExecutionProvider`, requires the `onnxruntime-gpu`
   package (see below).
3. Otherwise it falls back to plain CPU.

When a model loads you can see which mode engaged on stderr, e.g.
`[device] PixAI Tagger v0.9: GPU acceleration active (providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'])`
or `... no usable GPU detected - running on CPU ...`.

Force CPU only (even when a GPU is available) with `--cpu`.

### NVIDIA / Google Colab

The default `onnxruntime` package is CPU-only; for NVIDIA GPUs install the GPU
build. On Google Colab (CUDA 12 + cuDNN 12) or a local CUDA-12.x machine:

```
pip uninstall -y onnxruntime onnxruntime-gpu
pip install -r requirements-gpu.txt        # pins onnxruntime-gpu==1.20.2
# or just: pip install onnxruntime-gpu==1.20.2
```

> `onnxruntime` and `onnxruntime-gpu` share the same Python module and **must
> not** be installed at the same time - uninstalling first (as above) avoids a
> silent fallback to CPU.

Make sure a GPU runtime is selected: Colab -> Runtime -> Change runtime type ->
hardware accelerator **GPU**. Verify CUDA is visible to onnxruntime:

```python
import onnxruntime as ort
print(ort.get_available_providers())   # must include 'CUDAExecutionProvider'
```

If `CUDAExecutionProvider` is missing or you hit a CUDA/cuDNN library version
mismatch, try a newer build (`pip install onnxruntime-gpu` without a pin), or
reinstall the CPU one (`pip install onnxruntime==1.20.2`).

https://onnxruntime.ai/docs/install/</br>
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

## tests

Unit and integration tests live in `tests/` (requires `pytest`):

```
pip install pytest
python -m pytest tests/ -v
```

Integration tests that need a downloaded model are skipped automatically when
the model is not yet in the Hugging Face cache.

## Copyright

Public domain, except borrowed parts (e.g. `dbimutils.py`)
