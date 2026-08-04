import sys
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from tagger.interrogator.interrogator import AbsInterrogator

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / 'run.py'


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(RUN_PY), *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


class TestPostprocessTagsRegression:
    """The shared postprocess helper must keep working for all models."""

    def test_threshold_and_sort(self):
        out = AbsInterrogator.postprocess_tags(
            {'a': 0.9, 'b': 0.1, 'c': 0.5}, threshold=0.35)
        assert list(out) == ['a', 'c']

    def test_additional_tags_force_1(self):
        out = AbsInterrogator.postprocess_tags(
            {'a': 0.5}, threshold=0.9, additional_tags=['extra'])
        assert out['extra'] == 1.0

    def test_exclude_tags(self):
        out = AbsInterrogator.postprocess_tags(
            {'a': 0.9, 'b': 0.9}, threshold=0.1, exclude_tags=['b'])
        assert 'b' not in out

    def test_escape_and_underscore(self):
        out = AbsInterrogator.postprocess_tags(
            {'tag_name_(x)': 0.9}, threshold=0.0,
            escape_tag=True, replace_underscore=True)
        assert list(out) == [r'tag name \(x\)']

    def test_zero_threshold_keeps_everything(self):
        # pixai path passes threshold=0; everything through postprocess survives
        out = AbsInterrogator.postprocess_tags({'x': 0.01, 'y': 0.99},
                                               threshold=0.0)
        assert set(out) == {'x', 'y'}


class TestCLIParsing:
    def test_help_lists_new_flags(self):
        r = run_cli('-h')
        assert r.returncode == 0
        assert '--general-threshold' in r.stdout
        assert '--character-threshold' in r.stdout
        assert '--threshold' in r.stdout

    def test_threshold_default_is_none(self):
        # parse_args with no --threshold should leave args.threshold None so
        # that pixai can fall back to its per-category defaults
        import importlib
        sys.argv = ['run.py', '--file', 'x.png']
        spec = importlib.util.spec_from_file_location('run', RUN_PY)
        mod = importlib.util.module_from_spec(spec)
        # stub the interrogator so the module-level file path is never used
        fake_itg = type('F', (), {'interrogate': lambda self, im: ({}, {'a': 1.0}),
                                  'use_cpu': lambda self: None})()
        import tagger.interrogators as ti
        orig = ti.interrogators['wd14-convnextv2.v1']
        ti.interrogators['wd14-convnextv2.v1'] = fake_itg
        # make --file point at a real (temporary) image so module-level code runs
        tmp_img = Path(self._make_tmp_image())
        sys.argv = ['run.py', '--file', str(tmp_img)]
        try:
            spec.loader.exec_module(mod)
            assert mod.args.threshold is None
            assert mod.effective_postprocess_threshold == pytest.approx(0.35)
            assert not mod.pixai_mode
        finally:
            ti.interrogators['wd14-convnextv2.v1'] = orig
            tmp_img.unlink(missing_ok=True)

    @staticmethod
    def _make_tmp_image(suffix='.png'):
        import tempfile
        fd, path = tempfile.mkstemp(suffix=suffix)
        Image.new('RGB', (8, 8), (0, 0, 0)).save(path)
        return path

    def test_pixai_model_sets_pixai_mode(self):
        import importlib
        tmp_img = Path(self._make_tmp_image())
        # stub pixai interrogator so no real inference happens at import time
        import tagger.interrogators as ti
        from tagger.interrogator.pixaitaggerinterrogator import PixAITaggerInterrogator
        orig = ti.interrogators['pixai-tagger-v0.9']
        fake = PixAITaggerInterrogator('t', repo_id='dummy')
        fake.interrogate = lambda im: ({}, {})
        ti.interrogators['pixai-tagger-v0.9'] = fake
        sys.argv = ['run.py', '--file', str(tmp_img), '--model', 'pixai-tagger-v0.9']
        spec = importlib.util.spec_from_file_location('run', RUN_PY)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            assert mod.pixai_mode is True
            assert mod.effective_postprocess_threshold == 0.0
            # defaults wired into the interrogator: general 0.3, char 0.85
            assert fake.general_threshold == pytest.approx(0.3)
            assert fake.character_threshold == pytest.approx(0.85)
        finally:
            ti.interrogators['pixai-tagger-v0.9'] = orig
            tmp_img.unlink(missing_ok=True)
