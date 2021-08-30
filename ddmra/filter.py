"""Functions for the Earl filter."""
import math

import numpy as np
from scipy import signal

from .utils import get_fd_power


def respiration_iirnotch(TR_in_sec, bpm_min=18.582, bpm_max=25.7263):
    """Calculate filter parameters for respiration filter.

    Takes in the TR (optional min/max breaths-per-min, bpm_min, bpm_max).
    Returns the parameters for IIR Notch filter.
    """
    fs = 1.0 / TR_in_sec  # Sampling frequency (Hz)
    fn = fs / 2.0  # Nyquist frequency (Hz)

    # RR MIN
    rr_min = bpm_min / 60.0  # respiration rate minimum in Hz
    fa_min = abs(rr_min - math.floor((rr_min + fn) / fs) * fs)  # Aliased minimum frequency (Hz)
    w0_min = fa_min / fn  # Normalized minimum frequency

    # RR MAX
    rr_max = bpm_max / 60.0  # respiration rate maximum in Hz
    fa_max = abs(rr_max - math.floor((rr_max + fn) / fs) * fs)  # Aliased maximum frequency (Hz)
    w0_max = fa_max / fn  # Normalized maximum frequency

    # RR iirnotch filter
    w0 = np.mean([w0_min, w0_max])  # Mean normalized frequency
    bw = abs(w0_max - w0_min)  # Normalized bandwidth
    Q = w0 / bw  # Quality factor
    b, a = signal.iirnotch(w0, Q)  # Filter design

    return b, a


def filter_earl(motpars, t_r, radius=50):
    """Apply Earl filter to motion parameters.

    Parameters
    ----------
    motpars : (T, 6) array_like
        Raw motion parameters. First three columns are translations and last
        three are rotations. Rotations in degrees.
    t_r : :obj:`float`
        Repetition time in seconds.
    radius : :obj:`float`, optional
        Head radius for conversion of rotation units to distance at the edge of the brain.

    Returns
    -------
    motpars_filtered : (T, 6) array_like
    fd_filtered : (T,) array_like
    """
    # Create the filter
    b, a = respiration_iirnotch(t_r, bpm_min=18.582, bpm_max=25.7263)

    # Filter motion numbers
    filt_moco2 = signal.filtfilt(b, a, motpars, axis=0, padtype=None)
    filt_moco = signal.filtfilt(b, a, filt_moco2, axis=0, padtype=None)

    # Calculate FD values for the series
    fd = get_fd_power(filt_moco, order=["x", "y", "z", "r", "p", "ya"], unit="rad", radius=radius)

    return filt_moco, fd
