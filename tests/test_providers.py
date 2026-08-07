import sys
from unittest.mock import MagicMock, patch

import pytest

from tagger.interrogator.interrogator import (
    AbsInterrogator,
    ACCELERATOR_PROVIDERS,
)


def fresh(**kwargs) -> AbsInterrogator:
    """A minimal concrete AbsInterrogator for provider-logic tests."""
    return AbsInterrogator('Test Tagger')


class TestAcceleratorDetection:
    @pytest.mark.parametrize('providers,expected', [
        (['CUDAExecutionProvider', 'CPUExecutionProvider'], True),
        (['CoreMLExecutionProvider', 'CPUExecutionProvider'], True),
        (['TensorrtExecutionProvider'], True),
        (['CPUExecutionProvider'], False),
        (['CPUExecutionProvider', 'AzureExecutionProvider'], False),
        ([], False),
    ])
    def test_uses_accelerator(self, providers, expected):
        itg = fresh()
        assert itg.uses_accelerator(providers) is expected

    def test_uses_accelerator_defaults_to_self_providers(self):
        itg = fresh()
        itg.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        assert itg.uses_accelerator() is True
        itg.providers = ['CPUExecutionProvider']
        assert itg.uses_accelerator() is False

    def test_known_accelerators_are_recognized(self):
        for p in ['CUDAExecutionProvider', 'CoreMLExecutionProvider']:
            assert p in ACCELERATOR_PROVIDERS


class TestOptimalProviderSelection:
    def test_prefers_coreml_when_available(self):
        with patch.object(AbsInterrogator, 'get_available_providers',
                          return_value=['CoreMLExecutionProvider',
                                        'AzureExecutionProvider',
                                        'CPUExecutionProvider']):
            sel = fresh().providers
        assert sel[0] == 'CoreMLExecutionProvider'
        assert 'CPUExecutionProvider' in sel

    def test_prefers_cuda_when_no_coreml(self):
        with patch.object(AbsInterrogator, 'get_available_providers',
                          return_value=['CUDAExecutionProvider',
                                        'CPUExecutionProvider']):
            sel = fresh().providers
        assert sel == ['CUDAExecutionProvider', 'CPUExecutionProvider']

    def test_falls_back_to_cpu_only(self):
        with patch.object(AbsInterrogator, 'get_available_providers',
                          return_value=['CPUExecutionProvider']):
            sel = fresh().providers
        assert sel == ['CPUExecutionProvider']
        assert fresh().providers  # selection is never empty

    def test_cpu_forces_cpu_providers(self):
        itg = fresh()
        itg.use_cpu()
        assert itg.providers == ['CPUExecutionProvider']
        assert itg.uses_accelerator() is False


class TestProviderModeLogging:
    def _with_actual(self, itg, actual_providers):
        """Attach a fake model whose get_providers() returns actual_providers."""
        model = MagicMock()
        model.get_providers.return_value = actual_providers
        itg.model = model
        return itg

    def test_gpu_message_on_cuda(self, capsys):
        itg = self._with_actual(fresh(),
                                ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        itg.log_provider_mode()
        err = capsys.readouterr().err
        assert 'GPU acceleration active' in err
        assert 'CUDAExecutionProvider' in err

    def test_gpu_message_on_coreml(self, capsys):
        itg = self._with_actual(fresh(),
                                ['CoreMLExecutionProvider', 'CPUExecutionProvider'])
        itg.log_provider_mode()
        assert 'GPU acceleration active' in capsys.readouterr().err

    def test_cpu_fallback_message(self, capsys):
        itg = self._with_actual(fresh(), ['CPUExecutionProvider'])
        itg.log_provider_mode()
        err = capsys.readouterr().err
        assert 'no usable GPU detected' in err
        assert 'onnxruntime-gpu' in err

    def test_falls_back_to_requested_providers_when_no_model(self, capsys):
        itg = fresh()
        if hasattr(itg, 'model'):
            del itg.model
        itg.providers = ['CPUExecutionProvider']
        itg.log_provider_mode()
        assert 'no usable GPU detected' in capsys.readouterr().err
