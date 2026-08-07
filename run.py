import sys

from typing import Generator, Iterable
from tagger.interrogator.interrogator import AbsInterrogator
from PIL import Image, ImageFile
from pathlib import Path
import argparse

from tagger.interrogator.pixaitaggerinterrogator import PixAITaggerInterrogator
from tagger.interrogators import interrogators

# Allow images with broken headers to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

_EPILOGUE = """\
Available --model values:

  recommended for LoRA captioning:
    pixai-tagger-v0.9            PixAI Tagger v0.9 (2025, Danbooru snapshot
                                 2025-01). Per-category thresholds: general
                                 tags 0.3, character tags 0.85 (override with
                                 --general-threshold / --character-threshold).
                                 Detected characters also add their series tag.
    wd-eva02-large-tagger-v3     best SmilingWolf model (~0.3B params, 2024)
    camie-tagger-v2              70k tags, great character recognition (2025)

  SmilingWolf WD14 family:
    wd-vit-large-tagger-v3       large v3 (2024)
    wd-v1-4-vit-tagger.v3        v3 (2024)
    wd-v1-4-convnext-tagger.v3   v3 (2024)
    wd-v1-4-swinv2-tagger.v3     v3 (2024)
    wd14-vit.v2                  v2 (2023)
    wd14-convnext.v2             v2 (2023)
    wd-v1-4-moat-tagger.v2       v2 (2023)
    wd14-vit.v1                  v1 (2022)
    wd14-convnext.v1             v1 (2022)
    wd14-convnextv2.v1           v1 (default, 2023)
    wd14-swinv2-v1               v1 (2023)

  other:
    camie-tagger                 Camie Tagger v1 (2025)
    z3d-e621-convnext-toynya     e621/furry tags
    z3d-e621-convnext-silveroxides  e621/furry tags
    mld-caformer.dec-5-97527     ML-Danbooru (coarse generic tags only)
    mld-tresnetd.6-30000         ML-Danbooru (coarse generic tags only)

Examples:
  # tag one image with the default model (WebUI-style tags to stdout)
  python run.py --file image.jpg

  # tag a whole directory with PixAI (writes one .txt caption per image)
  python run.py --model pixai-tagger-v0.9 --dir training_images

  # LoRA dataset: add your trigger tag, drop noise tags, overwrite old captions
  python run.py --model pixai-tagger-v0.9 --dir training_images \\
      --additional-tag "my_trigger_tag" --exclude-tag "watermark,signature" \\
      --overwrite

  # keep exact Danbooru spelling (underscores, parentheses unescaped)
  python run.py --model pixai-tagger-v0.9 --file image.jpg --rawtag

Notes:
  With --dir, an existing caption file is kept by default; use --overwrite to
  regenerate. Supported inputs: .png, .jpg, .jpeg, .webp (add --recursive to
  include subfolders). All other models filter tags with --threshold (0.35);
  pixai uses its per-category thresholds instead.
"""

parser = argparse.ArgumentParser(
    description='Tag anime-style images with Danbooru-style tags and write '
                'one caption .txt next to each image (--dir mode).',
    epilog=_EPILOGUE,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--dir', help='Predictions for all images in the directory')
group.add_argument('--file', help='Predictions for one file')

parser.add_argument(
    '--threshold',
    type=float,
    default=None,
    help='Prediction threshold (default is 0.35; for PixAI: general 0.3, character 0.85)')
parser.add_argument(
    '--general-threshold',
    dest='general_threshold',
    type=float,
    default=None,
    help='PixAI only: threshold for general tags (default is 0.3; overrides --threshold)')
parser.add_argument(
    '--character-threshold',
    dest='character_threshold',
    type=float,
    default=None,
    help='PixAI only: threshold for character tags (default is 0.85)')
parser.add_argument(
    '--ext',
    default='.txt',
    help='Extension to add to caption file in case of dir option (default is .txt)')
parser.add_argument(
    '--overwrite',
    action='store_true',
    help='Overwrite caption file if it exists')
parser.add_argument(
    '--cpu',
    action='store_true',
    help='Use CPU only')
parser.add_argument(
    '--rawtag',
    action='store_true',
    help='Use the raw output of the model')
parser.add_argument(
    '--recursive',
    action='store_true',
    help='Enable recursive file search')
parser.add_argument(
    '--exclude-tag',
    dest='exclude_tags',
    action='append',
    metavar='t1,t2,t3',
    help='Specify tags to exclude (Need comma-separated list)')
parser.add_argument(
    '--additional-tag',
    dest='additional_tags',
    action='append',
    metavar='t1,t2,t3',
    help='Specify additional tags (Need comma-separated list)')
parser.add_argument(
    '--model',
    default='wd14-convnextv2.v1',
    choices=sorted(interrogators.keys()),
    metavar='MODELNAME',
    help='modelname to use for prediction (default is wd14-convnextv2.v1; '
         'see full list below)')
args = parser.parse_args()

# get interrogator configs
interrogator = interrogators[args.model]

if args.cpu:
    interrogator.use_cpu()

# PixAI applies its own per-category thresholds inside the interrogator;
# configure them here and skip the second (global) threshold filter later.
pixai_mode = isinstance(interrogator, PixAITaggerInterrogator)
if pixai_mode:
    interrogator.set_thresholds(
        general=args.general_threshold if args.general_threshold is not None
                else args.threshold,
        character=args.character_threshold,
    )
    # general_threshold defaults to the model default (0.3) unless the user
    # passed --threshold/--general-threshold; character defaults to 0.85.
    effective_postprocess_threshold = 0.0
else:
    effective_postprocess_threshold = (
        args.threshold if args.threshold is not None else 0.35
    )

def parse_exclude_tags() -> set[str]:
    if args.exclude_tags is None:
        return set()

    tags = []
    for str in args.exclude_tags:
        for tag in str.split(','):
            tags.append(tag.strip())

    # reverse escape (nai tag to danbooru tag)
    reverse_escaped_tags = []
    for tag in tags:
        tag = tag.replace(' ', '_').replace(r'\(', '(').replace(r'\)', ')')
        reverse_escaped_tags.append(tag)
    return set([*tags, *reverse_escaped_tags])  # reduce duplicates

def parse_additional_tags() -> list[str]:
    if args.additional_tags is None:
        return list()

    tags = []
    for str in args.additional_tags:
        for tag in str.split(','):
            tags.append(tag.strip())
    return list(set(tags))

def image_interrogate(image_path: Path, tag_escape: bool, exclude_tags: Iterable[str], additional_tags: list[str]) -> dict[str, float]:
    """
    Predictions from a image path
    """
    im = Image.open(image_path)
    result = interrogator.interrogate(im)

    return AbsInterrogator.postprocess_tags(
        result[1],
        threshold=effective_postprocess_threshold,
        escape_tag=tag_escape,
        replace_underscore=tag_escape,
        exclude_tags=exclude_tags,
        additional_tags=additional_tags)

def explore_image_files(folder_path: Path) -> Generator[Path, None, None]:
    """
    Explore files by folder path
    """
    for path in folder_path.iterdir():
        if path.is_file() and path.suffix in ['.png', '.jpg', '.jpeg', '.webp']:
            yield path
        elif args.recursive and path.is_dir():
            yield from explore_image_files(path)

if args.dir:
    root_path = Path(args.dir)
    for image_path in explore_image_files(root_path):
        caption_path = image_path.parent / f'{image_path.stem}{args.ext}'

        if caption_path.is_file() and not args.overwrite:
            # skip if caption exists
            print('skip:', image_path)
            continue

        print('processing:', image_path)
        tags = image_interrogate(image_path, not args.rawtag, parse_exclude_tags(), parse_additional_tags())

        tags_str = ', '.join(tags.keys())

        with open(caption_path, 'w') as fp:
            fp.write(tags_str)

if args.file:
    tags = image_interrogate(Path(args.file), not args.rawtag, parse_exclude_tags(), parse_additional_tags())
    print(file=sys.stderr)
    tags_str = ', '.join(tags.keys())
    print(tags_str)


