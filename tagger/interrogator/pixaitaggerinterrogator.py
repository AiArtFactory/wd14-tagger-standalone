import os
import json
import sys
import numpy as np
import pandas as pd

from typing import Tuple, Set
from PIL import Image

from pathlib import Path
from huggingface_hub import hf_hub_download

from tagger.interrogator.interrogator import AbsInterrogator

# Danbooru-style tag category ids used by the pixai-tagger onnx export
GENERAL_CATEGORY = 0
CHARACTER_CATEGORY = 4

# PixAI's recommended default thresholds (see deepghs/pixai-tagger-v0.9-onnx thresholds.csv)
DEFAULT_GENERAL_THRESHOLD = 0.3
DEFAULT_CHARACTER_THRESHOLD = 0.85


class PixAITaggerInterrogator(AbsInterrogator):
    """Interrogator for pixai-labs/pixai-tagger-v0.9 (ONNX export by deepghs).

    Unlike the WD14 models, the pixai tagger has no rating classes. Its tag
    table is split into two categories: general tags and character tags, each
    with its own recommended threshold. Detected character tags can also be
    expanded into their series (IP) tags using the ``ips`` column of the
    tag table (mirroring the ``char_ip_map.json`` behaviour of the upstream
    PyTorch handler).
    """

    def __init__(
        self,
        name: str,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        add_series_tags=True,
        **kwargs
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.kwargs = kwargs
        # Whether to add the series (IP) tag for each detected character tag.
        self.add_series_tags = add_series_tags
        # Per-category thresholds, falling back to the single global threshold.
        self.general_threshold = DEFAULT_GENERAL_THRESHOLD
        self.character_threshold = DEFAULT_CHARACTER_THRESHOLD
        self.character_tags: Set[str] = set()
        self.character_to_ips: dict[str, list[str]] = {}

    def set_thresholds(
        self,
        general: float = None,
        character: float = None
    ) -> None:
        """Optionally override the per-category thresholds."""
        if general is not None:
            self.general_threshold = general
        if character is not None:
            self.character_threshold = character

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}", file=sys.stderr)

        model_path = Path(hf_hub_download(
            **self.kwargs, filename=self.model_path))
        tags_path = Path(hf_hub_download(
            **self.kwargs, filename=self.tags_path))
        return model_path, tags_path

    def load(self) -> None:
        model_path, tags_path = self.download()

        from onnxruntime import InferenceSession
        self.model = InferenceSession(str(model_path), providers=self.providers)

        print(f'Loaded {self.name} model from {model_path}', file=sys.stderr)
        self.log_provider_mode()

        self.tags = pd.read_csv(tags_path)

        # index the character tags for fast per-category thresholding and
        # build the character -> series (IP) lookup from the ``ips`` column
        character_rows = self.tags[self.tags['category'] == CHARACTER_CATEGORY]
        self.character_tags = set(character_rows['name'])
        self.character_to_ips = {
            row['name']: json.loads(row['ips'])
            for _, row in character_rows.iterrows()
            if isinstance(row['ips'], str) and row['ips'] != '[]'
        }

    def preprocess(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Preprocess for the deepghs ONNX export (NCHW, normalized).

        The ONNX graph consumes an already-normalized, channels-first tensor
        (``[batch, 3, 448, 448]``) and returns a sigmoid-ed ``prediction``
        output. Resize + ``(x - 0.5) / 0.5`` normalization therefore happen in
        Python here, matching the repo's ``preprocess.json``.
        """
        # PIL RGB -> resize to model input, /255 -> (x-0.5)/0.5, HWC -> CHW
        image = np.asarray(
            Image.fromarray(image).resize((target_size, target_size), Image.BILINEAR),
            dtype=np.float32,
        )
        image = image / 255.0
        image = (image - 0.5) / 0.5
        image = image.transpose(2, 0, 1)          # HWC -> CHW
        image = np.expand_dims(image, 0)          # add batch dim
        return image

    def interrogate(
        self,
        input_image: Image.Image
    ) -> tuple[
        dict[str, float],  # rating confidents (always empty for this model)
        dict[str, float]   # tag confidents
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        if self.model is None:
            raise Exception("Model not loading.")

        # convert an image to fit the model (channels-first ONNX export)
        _, _, height, _ = self.model.get_inputs()[0].shape

        # alpha to white
        image = input_image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        image = self.preprocess(image, height)

        # evaluate model; the 'prediction' output is already sigmoid-ed
        input_name = self.model.get_inputs()[0].name
        label_name = 'prediction' if any(
            o.name == 'prediction' for o in self.model.get_outputs()
        ) else self.model.get_outputs()[0].name
        confidents = self.model.run([label_name], {input_name: image})[0]

        if self.tags is None:
            raise Exception("Tags not loading.")

        tags = self.tags[:][['name']]
        tags['confidents'] = confidents[0]

        tag_confidents = dict(tags.values)

        # keep only tags passing their per-category threshold
        tag_confidents = {
            name: conf
            for name, conf in tag_confidents.items()
            if conf >= (
                self.character_threshold
                if name in self.character_tags
                else self.general_threshold
            )
        }

        # expand character tags with their series (IP) tags. Series tags are
        # general tags, so they get the general tag confidence of the character
        # they were derived from.
        if self.add_series_tags:
            series_tags = {}
            for name, conf in tag_confidents.items():
                for ip in self.character_to_ips.get(name, []):
                    # keep the strongest confidence if several characters map
                    # to the same series
                    if ip not in series_tags or conf > series_tags[ip]:
                        series_tags[ip] = conf
            tag_confidents.update(series_tags)

        # no rating classes in this model
        return {}, tag_confidents
