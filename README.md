
forked from [https://github.com/picobyte/stable-diffusion-webui-wd14-tagger](https://github.com/picobyte/stable-diffusion-webui-wd14-tagger)

## install

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## usage

```
usage: run.py [-h] (--dir DIR | --file FILE) [--threshold THRESHOLD] [--ext EXT] [--overwrite] [--cpu] [--rawtag] [--recursive] [--exclude-tag t1,t2,t3] [--model MODELNAME]

options:
  -h, --help            show this help message and exit
  --dir DIR             Predictions for all images in the directory
  --file FILE           Predictions for one file
  --threshold THRESHOLD
                        Prediction threshold (default is 0.35)
  --ext EXT             Extension to add to caption file in case of dir option (default is .txt)
  --overwrite           Overwrite caption file if it exists
  --cpu                 Use CPU only
  --rawtag              Use the raw output of the model
  --recursive           Enable recursive file search
  --exclude-tag t1,t2,t3
                        Specify tags to exclude (Need comma-separated list)
  --additional-tag t1,t2,t3
                        Specify tags to append (Need comma-separated list)
  --model MODELNAME     modelname to use for prediction (default is wd14-convnextv2.v1)
```

single file

```
python run.py --file image.jpg
```

batch execution

```
python run.py --dir dir/dir
```

## Support Models

```
# PixAI Tagger v0.9 model. (released 2025, Danbooru snapshot 2025-01)
# Uses per-category thresholds: general tags 0.3, character tags 0.85.
# Character tags additionally add their series (IP) tag.
python run.py --model pixai-tagger-v0.9 --file image.jpg

# Camie Tagger / Camie Tagger v2 model.(released 2025)
python run.py --model camie-tagger --file image.jpg
python run.py --model camie-tagger-v2 --file image.jpg

# SmilingWolf large model. (released 2024)
python run.py --model wd-vit-large-tagger-v3 --file image.jpg
python run.py --model wd-eva02-large-tagger-v3 --file image.jpg

# SmilingWolf v3 model. (released 2024)
python run.py --model wd-v1-4-vit-tagger.v3 --file image.jpg
python run.py --model wd-v1-4-convnext-tagger.v3 --file image.jpg
python run.py --model wd-v1-4-swinv2-tagger.v3 --file image.jpg

# SmilingWolf v2 model. (released 2023)
python run.py --model wd-v1-4-moat-tagger.v2  --file image.jpg
python run.py --model wd14-vit.v2  --file image.jpg
python run.py --model wd14-convnext.v2  --file image.jpg

# SmilingWolf v1 model. (released 2022)
python run.py --model wd14-vit.v1 --file image.jpg
python run.py --model wd14-convnext.v1 --file image.jpg
python run.py --model wd14-convnextv2.v1 --file image.jpg
python run.py --model wd14-swinv2-v1 --file image.jpg

# Z3D-E621-Convnext
python run.py --model z3d-e621-convnext-toynya --file image.jpg
python run.py --model z3d-e621-convnext-silveroxides --file image.jpg

# kiriyamaX model.
python run.py --model mld-caformer.dec-5-97527 --file image.jpg
python run.py --model mld-tresnetd.6-30000 --file image.jpg
```

## PixAI Tagger per-category thresholds

The PixAI model applies its own recommended per-category thresholds internally
(general tags `0.3`, character tags `0.85`), and character tags automatically
contribute their series (IP) tag (e.g. `1girl, ..., artoria pendragon \(fate\), ...,
fate \(series\)`). Two extra flags can override them:

```
# give the character filter a looser threshold, general stays 0.3
python run.py --model pixai-tagger-v0.9 --dir images/ --character-threshold 0.5

# override the general-tag threshold (and/or use --threshold as a shorthand
# for the general threshold)
python run.py --model pixai-tagger-v0.9 --dir images/ --general-threshold 0.2
```

For all other models `--threshold` (default `0.35`) keeps its original meaning.

## Using GPU

Requires CUDA 12.2 and cuDNN8.x.

```
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

https://onnxruntime.ai/docs/install/</br>
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

## Copyright

Public domain, except borrowed parts (e.g. `dbimutils.py`)
