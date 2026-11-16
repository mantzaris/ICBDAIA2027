import numpy as np
import pytest

from acoustic_regeneration.agents.analysis_tools import AcousticTools


def clip():
    t = np.arange(32768) / 8000
    audio = np.sin(2 * np.pi * 78.125 * t)
    known = np.ones(len(audio), dtype=bool)
    known[10159:14159] = False
    return audio, known


def test_missing_reference_cannot_change_tool_measurements():
    audio, known = clip()
    changed = audio.copy()
    changed[~known] = np.nan
    left, right = AcousticTools(audio, known), AcousticTools(changed, known)
    for window in (32, 128):
        assert left.analyze_spectrum(window) == right.analyze_spectrum(window)
    assert left.audio_hash() == right.audio_hash()


def test_frequency_resolution_and_mask_exclusion():
    audio, known = clip()
    tools = AcousticTools(audio, known)
    short, long = tools.analyze_spectrum(32), tools.analyze_spectrum(128)
    assert short['bin_spacing_hz'] == 31.25 and long['bin_spacing_hz'] == 7.8125
    assert long['observed_peak_bin_hz'] == 78.125
    for window in (32, 128):
        _, _, powers, valid = tools.spectrum_arrays(window)
        assert np.isnan(powers[:, ~valid]).all()
        assert np.isfinite(powers[:, valid]).all()


def test_configuration_requires_evidence_and_preserves_audio():
    audio, known = clip()
    tools = AcousticTools(audio, known)
    with pytest.raises(ValueError, match='Analyze'):
        tools.configure_display(128)
    tools.analyze_spectrum(128)
    tools.configure_display(128)
    result = tools.verify_display()
    assert result['current_window_ms'] == 128
    assert result['audio_unchanged'] and result['missing_samples_remain_erased']
    assert not result['physical_hardware_changed']


def test_tool_boundary_rejects_unknown_actions_and_settings():
    audio, known = clip()
    tools = AcousticTools(audio, known)
    with pytest.raises(ValueError):
        tools.call('change_gain', {})
    with pytest.raises(ValueError):
        tools.analyze_spectrum(256)
    with pytest.raises(TypeError):
        tools.call('inspect_recording', {'source_path': '/arbitrary'})
