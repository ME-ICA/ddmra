"""Tests for the ddmra.filter module (Earl respiration filter)."""

import numpy as np

from ddmra import filter as ddmra_filter


def test_respiration_iirnotch_coefficients():
    """The notch filter returns finite, length-3 numerator/denominator arrays."""
    b, a = ddmra_filter.respiration_iirnotch(2.0)
    assert b.shape == (3,)
    assert a.shape == (3,)
    assert np.all(np.isfinite(b))
    assert np.all(np.isfinite(a))


def test_respiration_iirnotch_is_stable():
    """A valid IIR filter has all poles strictly inside the unit circle."""
    for t_r in (0.8, 1.5, 2.0, 3.0):
        _, a = ddmra_filter.respiration_iirnotch(t_r)
        poles = np.roots(a)
        assert np.all(np.abs(poles) < 1.0), f"Unstable filter for TR={t_r}"


def test_respiration_iirnotch_custom_band():
    """Custom breaths-per-minute bounds still produce a valid filter."""
    b, a = ddmra_filter.respiration_iirnotch(2.0, bpm_min=12.0, bpm_max=20.0)
    assert b.shape == (3,)
    assert np.all(np.isfinite(np.concatenate([b, a])))


def test_filter_earl_shapes_and_first_fd():
    """filter_earl returns filtered motion (T, 6) and FD (T,) with FD[0] == 0."""
    rng = np.random.RandomState(0)
    motpars = rng.normal(size=(100, 6))
    filtered, fd = ddmra_filter.filter_earl(motpars, t_r=2.0, radius=50)
    assert filtered.shape == motpars.shape
    assert fd.shape == (100,)
    assert np.all(np.isfinite(filtered))
    assert np.all(np.isfinite(fd))
    # The first FD value is always 0 (the derivative prepends a zero row).
    assert fd[0] == 0.0


def test_filter_earl_attenuates_respiration_band():
    """The filter should reduce power at the aliased respiration frequency."""
    t_r = 2.0
    n = 200
    t = np.arange(n) * t_r
    # Build a motion trace dominated by the respiration band the filter targets.
    # w0 is normalized to Nyquist; convert back to Hz to synthesize the signal.
    fs = 1.0 / t_r
    fn = fs / 2.0
    b, a = ddmra_filter.respiration_iirnotch(t_r)
    # Recover the notch center frequency (Hz) from the filter's frequency response.
    from scipy.signal import freqz

    w, h = freqz(b, a, worN=2048)
    notch_freq_hz = w[np.argmin(np.abs(h))] / np.pi * fn
    signal = np.sin(2 * np.pi * notch_freq_hz * t)
    motpars = np.tile(signal[:, None], (1, 6))
    filtered, _ = ddmra_filter.filter_earl(motpars, t_r=t_r)
    assert np.var(filtered) < np.var(motpars)
