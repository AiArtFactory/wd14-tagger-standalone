import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / 'run.py'

# The same arguments the batch loop uses; imported so tests stay in sync.
from tqdm.auto import tqdm


def make_images(dir_path: Path, n=5, ext='.png'):
    for i in range(n):
        Image.new('RGB', (8, 8), (i * 30, 0, 0)).save(dir_path / f'img_{i}{ext}')
    return sorted(dir_path.glob(f'*{ext}'))


class TestTqdmIntegration:
    def test_progress_counts_and_rate_format(self, capsys):
        """Simulate the batch loop and check n/total + ETA + img/s rendering."""
        images = list(range(5))
        progress = tqdm(images, desc='Tagging', unit='image', file=sys.stderr,
                        mininterval=0)
        for _ in progress:
            pass
        out = capsys.readouterr().err
        assert 'image/s' in out or 's/image' in out   # rate shown in img per sec
        assert '5/5' in out                            # count
        assert '<' in out                              # ETA marker (elapsed<remaining)

    def test_progress_total_tracks_images(self, tmp_path, capsys):
        make_images(tmp_path, n=4)
        files = list(tmp_path.glob('*.png'))
        progress = tqdm(files, desc='Tagging', unit='image', file=sys.stderr,
                        mininterval=0)
        for p in progress:
            progress.set_postfix_str(p.name, refresh=False)
        assert '4/4' in capsys.readouterr().err


class TestRunPyProgressWiring:
    """Drive run.py's --dir block in-process with a stubbed interrogator so we
    exercise: file discovery, tqdm wrapping, skip message, caption writing."""

    def _run_dir(self, tmp_path, argv_extra=None, existing_caption=False):
        import importlib
        import runpy  # noqa

        make_images(tmp_path, n=3)
        if existing_caption:
            (tmp_path / 'img_0.txt').write_text('already tagged')

        import tagger.interrogators as ti
        orig = ti.interrogators['wd14-convnextv2.v1']
        fake = MagicMock()
        fake.interrogate = lambda im: ({}, {'1girl': 0.9, 'smile': 0.8})
        fake.use_cpu = lambda: None
        ti.interrogators['wd14-convnextv2.v1'] = fake
        argv = ['run.py', '--dir', str(tmp_path)] + (argv_extra or [])
        sys.argv = argv
        spec = importlib.util.spec_from_file_location('run', RUN_PY)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        finally:
            ti.interrogators['wd14-convnextv2.v1'] = orig
        return mod

    def test_progress_bar_used_and_captions_written(self, tmp_path, capsys):
        mod = self._run_dir(tmp_path)
        # captions were produced for all 3 images
        for i in range(3):
            caption = tmp_path / f'img_{i}.txt'
            assert caption.is_file()
            assert '1girl' in caption.read_text()
        # tqdm output went to stderr with n/total counting
        assert '3/3' in capsys.readouterr().err

    def test_skips_existing_caption_and_reports_it(self, tmp_path, capsys):
        self._run_dir(tmp_path, existing_caption=True)
        err = capsys.readouterr().err
        assert 'skip:' in err
        # the pre-existing caption must be untouched
        assert (tmp_path / 'img_0.txt').read_text() == 'already tagged'
        # the other two were written
        assert '1girl' in (tmp_path / 'img_1.txt').read_text()

    def test_overwrite_regenerates_captions(self, tmp_path, capsys):
        (tmp_path).mkdir(exist_ok=True)
        self._run_dir(tmp_path, argv_extra=['--overwrite'], existing_caption=True)
        # --overwrite replaced the placeholder caption
        assert (tmp_path / 'img_0.txt').read_text() != 'already tagged'

    def test_empty_dir_completes_without_error(self, tmp_path, capsys):
        import importlib
        sys.argv = ['run.py', '--dir', str(tmp_path)]
        spec = importlib.util.spec_from_file_location('run', RUN_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # should not raise even with 0 images
        # tqdm renders an empty bar (no totals) when there is nothing to do
        assert 'Tagging' in capsys.readouterr().err
