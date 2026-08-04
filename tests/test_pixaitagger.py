import io
import json
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# run.py parses args at import time and requires --file/--dir, so stub out
# sys.argv before importing it in the CLI tests.

from tagger.interrogator.pixaitaggerinterrogator import (
    PixAITaggerInterrogator,
    GENERAL_CATEGORY,
    CHARACTER_CATEGORY,
    DEFAULT_GENERAL_THRESHOLD,
    DEFAULT_CHARACTER_THRESHOLD,
)
from tagger.interrogators import interrogators


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_tags_df():
    """A small fake tag table in the pixai format."""
    rows = [
        # id, tag_id, name, category, count, ips
        (0, 1, '1girl', GENERAL_CATEGORY, 1_000_000, '[]'),
        (1, 2, 'pink_hair', GENERAL_CATEGORY, 500_000, '[]'),
        (2, 3, 'hat', GENERAL_CATEGORY, 400_000, '[]'),
        (3, 4, 'artoria_pendragon_(fate)', CHARACTER_CATEGORY, 40_000,
         '["fate_(series)"]'),
        (4, 5, 'kousaka_tamaki', CHARACTER_CATEGORY, 2_000,
         '["to_heart_(series)", "to_heart_2"]'),
        (5, 6, 'unknown_solo_character', CHARACTER_CATEGORY, 1_000, '[]'),
    ]
    return pd.DataFrame(rows, columns=['id', 'tag_id', 'name', 'category', 'count', 'ips'])


def make_interrogator(confidences=None):
    """A PixAITaggerInterrogator with a stubbed-out ONNX model."""
    if confidences is None:
        confidences = [0.99, 0.5, 0.2, 0.9, 0.95, 0.91]

    itg = PixAITaggerInterrogator('PixAI Tagger v0.9 (test)', repo_id='dummy')

    # stub .tags/.category index the way load() does
    df = make_tags_df()
    itg.tags = df
    char_rows = df[df['category'] == CHARACTER_CATEGORY]
    itg.character_tags = set(char_rows['name'])
    itg.character_to_ips = {
        row['name']: json.loads(row['ips'])
        for _, row in char_rows.iterrows()
        if isinstance(row['ips'], str) and row['ips'] != '[]'
    }

    # fake onnx session
    model = MagicMock()
    inp = MagicMock()
    inp.shape = [1, 448, 448, 3]
    inp.name = 'input'
    model.get_inputs.return_value = [inp]
    out = MagicMock()
    out.name = 'output'
    model.get_outputs.return_value = [out]
    model.run.return_value = [np.array([confidences], dtype=np.float32)]
    itg.model = model
    return itg


def make_image(w=64, h=32, mode='RGBA'):
    im = Image.new(mode, (w, h), (123, 50, 200, 128))
    return im


# --------------------------------------------------------------------------
# unit tests: PixAITaggerInterrogator
# --------------------------------------------------------------------------

class TestTagIndexing:
    def test_character_tags_indexed(self):
        itg = make_interrogator()
        assert itg.character_tags == {
            'artoria_pendragon_(fate)', 'kousaka_tamaki', 'unknown_solo_character',
        }
        assert '1girl' not in itg.character_tags

    def test_character_to_ips_indexed(self):
        itg = make_interrogator()
        assert itg.character_to_ips == {
            'artoria_pendragon_(fate)': ['fate_(series)'],
            'kousaka_tamaki': ['to_heart_(series)', 'to_heart_2'],
        }

    def test_empty_character(self):
        itg = make_interrogator()
        assert 'unknown_solo_character' not in itg.character_to_ips


class TestThresholdFiltering:
    def test_default_thresholds(self):
        itg = make_interrogator()
        _, tags = itg.interrogate(make_image())
        # 1girl 0.99 -> general >= 0.3: kept
        # pink_hair 0.5 -> general: kept
        # hat 0.2 -> general: dropped (< 0.3)
        # artoria 0.9 -> character >= 0.85: kept
        # kousaka 0.95 -> character: kept
        # unknown_solo_character 0.91 -> character: kept
        assert '1girl' in tags
        assert 'pink_hair' in tags
        assert 'hat' not in tags
        assert 'artoria_pendragon_(fate)' in tags
        assert 'kousaka_tamaki' in tags
        assert 'unknown_solo_character' in tags

    def test_character_threshold_stricter_than_general(self):
        # a borderline character (0.6) passes general (0.3) but not char (0.85)
        itg = make_interrogator(confidences=[0.99, 0.5, 0.2, 0.6, 0.95, 0.91])
        _, tags = itg.interrogate(make_image())
        assert 'artoria_pendragon_(fate)' not in tags  # 0.6 < 0.85
        assert 'kousaka_tamaki' in tags

    def test_set_thresholds_override(self):
        itg = make_interrogator(confidences=[0.99, 0.5, 0.2, 0.5, 0.95, 0.91])
        itg.set_thresholds(general=0.1, character=0.4)
        _, tags = itg.interrogate(make_image())
        # hat 0.2 >= 0.1 now kept; artoria 0.5 >= 0.4 now kept
        assert 'hat' in tags
        assert 'artoria_pendragon_(fate)' in tags

    def test_set_thresholds_partial(self):
        itg = make_interrogator()
        itg.set_thresholds(character=0.999)  # raise only char threshold
        _, tags = itg.interrogate(make_image())
        assert '1girl' in tags  # general untouched
        assert 'artoria_pendragon_(fate)' not in tags

    def test_no_rating_output(self):
        itg = make_interrogator()
        ratings, _ = itg.interrogate(make_image())
        assert ratings == {}


class TestSeriesTags:
    def test_series_added_from_ips(self):
        itg = make_interrogator()
        _, tags = itg.interrogate(make_image())
        # fate_(series) from artoria; to_heart_* from kousaka
        assert 'fate_(series)' in tags
        assert 'to_heart_(series)' in tags
        assert 'to_heart_2' in tags

    def test_series_confidence_from_strongest_character(self):
        itg = make_interrogator()
        _, tags = itg.interrogate(make_image())
        # only artoria (0.9) maps to fate_(series)
        assert tags['fate_(series)'] == pytest.approx(0.9)

    def test_series_disabled(self):
        itg = make_interrogator()
        itg.add_series_tags = False
        _, tags = itg.interrogate(make_image())
        assert 'fate_(series)' not in tags
        assert 'artoria_pendragon_(fate)' in tags

    def test_series_not_duplicated_if_already_detected(self):
        # pretend 1girl detection also yields fate_(series) as a general tag
        names = ['1girl', 'pink_hair', 'fate_(series)']
        df = pd.DataFrame(
            [(0, 1, '1girl', GENERAL_CATEGORY, 1, '[]'),
             (1, 2, 'pink_hair', GENERAL_CATEGORY, 1, '[]'),
             (2, 3, 'fate_(series)', GENERAL_CATEGORY, 1, '[]'),
             (3, 4, 'artoria_pendragon_(fate)', CHARACTER_CATEGORY, 1,
              '["fate_(series)"]')],
            columns=['id', 'tag_id', 'name', 'category', 'count', 'ips'])
        itg = make_interrogator(confidences=[0.99, 0.5, 0.8, 0.9])
        itg.tags = df
        char_rows = df[df['category'] == CHARACTER_CATEGORY]
        itg.character_tags = set(char_rows['name'])
        itg.character_to_ips = {'artoria_pendragon_(fate)': ['fate_(series)']}
        model = itg.model
        model.run.return_value = [np.array([[0.99, 0.5, 0.8, 0.9]], dtype=np.float32)]
        _, tags = itg.interrogate(make_image())
        # character-derived confidence (0.9) beats detected general (0.8)
        assert tags['fate_(series)'] == pytest.approx(0.9)


class TestModelWiring:
    def test_registry_contains_pixai(self):
        assert 'pixai-tagger-v0.9' in interrogators
        assert isinstance(interrogators['pixai-tagger-v0.9'],
                          PixAITaggerInterrogator)

    def test_registry_points_to_onnx_repo(self):
        itg = interrogators['pixai-tagger-v0.9']
        assert itg.kwargs['repo_id'] == 'deepghs/pixai-tagger-v0.9-onnx'
        assert itg.model_path == 'model.onnx'
        assert itg.tags_path == 'selected_tags.csv'

    def test_default_thresholds_constants(self):
        assert DEFAULT_GENERAL_THRESHOLD == pytest.approx(0.3)
        assert DEFAULT_CHARACTER_THRESHOLD == pytest.approx(0.85)

    def test_existing_models_unchanged(self):
        # make sure the registry still has the original 18 models
        assert len(interrogators) == 19
        for name in ['wd14-convnextv2.v1', 'camie-tagger-v2',
                     'z3d-e621-convnext-toynya', 'mld-caformer.dec-5-97527']:
            assert name in interrogators


# --------------------------------------------------------------------------
# integration test: real model (skipped when not downloaded)
# --------------------------------------------------------------------------

def _model_cached():
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache('deepghs/pixai-tagger-v0.9-onnx', 'model.onnx')
        t = try_to_load_from_cache('deepghs/pixai-tagger-v0.9-onnx', 'selected_tags.csv')
        return p is not None and t is not None
    except Exception:
        return False


@pytest.mark.skipif(not _model_cached(),
                    reason='pixai ONNX model not in HF cache')
class TestRealModelIntegration:
    def test_end_to_end_synthetic_image(self):
        itg = interrogators['pixai-tagger-v0.9']
        itg.use_cpu()
        itg.unload()
        # a big flat-color image -> should at least predict common tags
        im = Image.new('RGB', (256, 256), (128, 128, 128))
        ratings, tags = itg.interrogate(im)
        assert ratings == {}
        # with 9k+ general tags even a blank image fires *something*
        assert isinstance(tags, dict)
        for v in tags.values():
            assert 0 <= v <= 1
        itg.unload()
