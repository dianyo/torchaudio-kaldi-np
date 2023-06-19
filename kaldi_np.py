import math
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "get_mel_banks",
    "inverse_mel_scale",
    "inverse_mel_scale_scalar",
    "mel_scale",
    "mel_scale_scalar",
    "spectrogram",
    "fbank",
    "mfcc",
    "vtln_warp_freq",
    "vtln_warp_mel_freq",
]

# numeric_limits<float>::epsilon() 1.1920928955078125e-07
EPSILON = np.finfo(np.float32).eps
# 1 milliseconds = 0.001 seconds
MILLISECONDS_TO_SECONDS = 0.001

# window types
HAMMING = "hamming"
HANNING = "hanning"
POVEY = "povey"
RECTANGULAR = "rectangular"
BLACKMAN = "blackman"
WINDOWS = [HAMMING, HANNING, POVEY, RECTANGULAR, BLACKMAN]


def _get_epsilon(dtype):
    return EPSILON.astype(dtype=dtype)


def _next_power_of_2(x: int) -> int:
    r"""Returns the smallest power of 2 that is greater than x"""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _get_strided(waveform: np.ndarray, window_size: int, window_shift: int, snip_edges: bool) -> np.ndarray:
    """
    Given a waveform (1D array of size `num_samples`), it returns a 2D array (m, `window_size`) representing
    how the window is shifted along the waveform. Each row is a frame.

    Args:
        waveform (ndarray): Array of size `num_samples`
        window_size (int): Frame length
        window_shift (int): Frame shift
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length. If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends.

    Returns:
        ndarray: 2D array of size (m, `window_size`) where each row is a frame
    """
    assert waveform.ndim == 1
    num_samples = waveform.shape[0]
    strides = (window_shift * waveform.strides[0], waveform.strides[0])

    if snip_edges:
        if num_samples < window_size:
            return np.empty((0, 0), dtype=waveform.dtype)
        else:
            m = 1 + (num_samples - window_size) // window_shift
    else:
        reversed_waveform = np.flip(waveform, axis=0)
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        pad_right = reversed_waveform
        if pad > 0:
            pad_left = reversed_waveform[-pad:]
            waveform = np.concatenate((pad_left, waveform, pad_right), axis=0)
        else:
            waveform = np.concatenate((waveform[-pad:], pad_right), axis=0)

    sizes = (m, window_size)
    return np.lib.stride_tricks.as_strided(waveform, sizes, strides)

def _feature_window_function(
    window_type: str,
    window_size: int,
    blackman_coeff: float,
) -> np.ndarray:
    r"""Returns a window function with the given type and size"""
    if window_type == HANNING:
        return np.hanning(window_size)
    elif window_type == HAMMING:
        return np.hamming(window_size)
    elif window_type == POVEY:
        # like hanning but goes to zero at edges
        return np.hanning(window_size) ** 0.85
    elif window_type == RECTANGULAR:
        return np.ones(window_size)
    elif window_type == BLACKMAN:
        a = 2 * math.pi / (window_size - 1)
        window_function = np.arange(window_size)
        return (
            blackman_coeff
            - 0.5 * np.cos(a * window_function)
            + (0.5 - blackman_coeff) * np.cos(2 * a * window_function)
        )
    else:
        raise Exception("Invalid window type " + window_type)


def _get_log_energy(strided_input: np.ndarray, epsilon: float, energy_floor: float) -> np.ndarray:
    """
    Returns the log energy of size (m) for a strided_input (m,*)
    """
    log_energy = np.log(np.maximum(np.sum(strided_input**2, axis=1), epsilon))  # size (m)
    if energy_floor == 0.0:
        return log_energy
    return np.maximum(log_energy, math.log(energy_floor))


def _get_waveform_and_window_properties(
    waveform: np.ndarray,
    channel: int,
    sample_frequency: float,
    frame_shift: float,
    frame_length: float,
    round_to_power_of_two: bool,
    preemphasis_coefficient: float
) -> Tuple[np.ndarray, int, int, int]:
    """
    Gets the waveform and window properties
    """
    channel = max(channel, 0)
    assert channel < waveform.shape[0], "Invalid channel {} for size {}".format(channel, waveform.shape[0])
    waveform = waveform[channel, :]  # size (n)
    window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
    window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
    padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size

    assert 2 <= window_size <= len(waveform), "Choose a window size {} that is [2, {}]".format(
        window_size, len(waveform)
    )
    assert 0 < window_shift, "`window_shift` must be greater than 0"
    assert padded_window_size % 2 == 0, (
        "The padded `window_size` must be divisible by two. "
        "Use `round_to_power_of_two` or change `frame_length`"
    )
    assert 0.0 <= preemphasis_coefficient <= 1.0, "`preemphasis_coefficient` must be between [0,1]"
    assert sample_frequency > 0, "`sample_frequency` must be greater than zero"
    return waveform, window_shift, window_size, padded_window_size


def _get_window(
    waveform: np.ndarray,
    padded_window_size: int,
    window_size: int,
    window_shift: int,
    window_type: str,
    blackman_coeff: float,
    snip_edges: bool,
    raw_energy: bool,
    energy_floor: float,
    dither: float,
    remove_dc_offset: bool,
    preemphasis_coefficient: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets a window and its log energy

    Returns:
        (np.ndarray, np.ndarray): strided_input of size (m, padded_window_size) and signal_log_energy of size (m)
    """
    dtype = waveform.dtype
    epsilon = _get_epsilon(dtype)

    # size (m, window_size)
    strided_input = _get_strided(waveform, window_size, window_shift, snip_edges)

    if dither != 0.0:
        rand_gauss = np.random.randn(*strided_input.shape)
        strided_input = strided_input + rand_gauss * dither

    if remove_dc_offset:
        # Subtract each row/frame by its mean
        row_means = np.mean(strided_input, axis=1).reshape(-1, 1)  # size (m, 1)
        strided_input = strided_input - row_means

    if raw_energy:
        # Compute the log energy of each row/frame before applying preemphasis and window function
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)  # size (m)

    if preemphasis_coefficient != 0.0:
        # strided_input[i,j] -= preemphasis_coefficient * strided_input[i, max(0, j-1)] for all i,j
        offset_strided_input = np.pad(strided_input, ((0, 0), (1, 0)), mode="edge")  # size (m, window_size + 1)
        strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :-1]

    # Apply window_function to each row/frame
    window_function = _feature_window_function(window_type, window_size, blackman_coeff).reshape(1, -1)  # size (1, window_size)
    strided_input = strided_input * window_function  # size (m, window_size)

    # Pad columns with zero until we reach size (m, padded_window_size)
    if padded_window_size != window_size:
        padding_right = padded_window_size - window_size
        strided_input = np.pad(strided_input, ((0, 0), (0, padding_right)), mode="constant", constant_values=0)

    # Compute energy after window function (not the raw one)
    if not raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)  # size (m)

    return strided_input, signal_log_energy


def _subtract_column_mean(tensor: np.ndarray, subtract_mean: bool) -> np.ndarray:
    # subtracts the column mean of the tensor size (m, n) if subtract_mean=True
    # it returns size (m, n)
    if subtract_mean:
        col_means = np.mean(tensor, axis=0, keepdims=True)
        tensor = tensor - col_means
    return tensor


def spectrogram(
    waveform: np.ndarray,
    blackman_coeff: float = 0.42,
    channel: int = -1,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    min_duration: float = 0.0,
    preemphasis_coefficient: float = 0.97,
    raw_energy: bool = True,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    subtract_mean: bool = False,
    window_type: str = POVEY,
) -> np.ndarray:
    r"""Create a spectrogram from a raw audio signal. This matches the input/output of Kaldi's
    compute-spectrogram-feats.

    Args:
        waveform (np.ndarray): Tensor of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff (float, optional): Constant coefficient for generalized Blackman window. (Default: ``0.42``)
        channel (int, optional): Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (Default: ``-1``)
        dither (float, optional): Dithering constant (0.0 means no dither). If you turn this off, you should set
            the energy_floor option, e.g. to 1.0 or 0.1 (Default: ``0.0``)
        energy_floor (float, optional): Floor on energy (absolute, not relative) in Spectrogram computation.  Caution:
            this floor is applied to the zeroth component, representing the total signal energy.  The floor on the
            individual spectrogram elements is fixed at std::numeric_limits<float>::epsilon(). (Default: ``1.0``)
        frame_length (float, optional): Frame length in milliseconds (Default: ``25.0``)
        frame_shift (float, optional): Frame shift in milliseconds (Default: ``10.0``)
        min_duration (float, optional): Minimum duration of segments to process (in seconds). (Default: ``0.0``)
        preemphasis_coefficient (float, optional): Coefficient for use in signal preemphasis (Default: ``0.97``)
        raw_energy (bool, optional): If True, compute energy before preemphasis and windowing (Default: ``True``)
        remove_dc_offset (bool, optional): Subtract mean from waveform on each frame (Default: ``True``)
        round_to_power_of_two (bool, optional): If True, round window size to power of two by zero-padding input
            to FFT. (Default: ``True``)
        sample_frequency (float, optional): Waveform data sample frequency (must match the waveform file, if
            specified there) (Default: ``16000.0``)
        snip_edges (bool, optional): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends. (Default: ``True``)
        subtract_mean (bool, optional): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (Default: ``False``)
        window_type (str, optional): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')
         (Default: ``'povey'``)

    Returns:
        np.ndarray: A spectrogram identical to what Kaldi would output. The shape is
        (m, ``padded_window_size // 2 + 1``) where m is calculated in _get_strided
    """
    dtype = waveform.dtype
    epsilon = _get_epsilon(dtype)

    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient
    )

    if len(waveform) < min_duration * sample_frequency:
        # signal is too short
        return np.empty((0,))

    strided_input, signal_log_energy = _get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        window_type,
        blackman_coeff,
        snip_edges,
        raw_energy,
        energy_floor,
        dither,
        remove_dc_offset,
        preemphasis_coefficient,
    )

    # size (m, padded_window_size // 2 + 1, 2)
    fft = np.fft.rfft(strided_input)

    # Convert the FFT into a power spectrum
    power_spectrum = np.log(np.maximum(np.abs(fft) ** 2, epsilon))
    power_spectrum[:, 0] = signal_log_energy

    power_spectrum = _subtract_column_mean(power_spectrum, subtract_mean)
    return power_spectrum


def inverse_mel_scale_scalar(mel_freq: float) -> float:
    return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)


def inverse_mel_scale(mel_freq: np.ndarray) -> np.ndarray:
    return 700.0 * (np.exp(mel_freq / 1127.0) - 1.0)


def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def mel_scale(freq: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + freq / 700.0)


def vtln_warp_freq(
    vtln_low_cutoff: float,
    vtln_high_cutoff: float,
    low_freq: float,
    high_freq: float,
    vtln_warp_factor: float,
    freq: np.ndarray,
) -> np.ndarray:
    r"""This computes a VTLN warping function that is not the same as HTK's one,
    but has similar inputs (this function has the advantage of never producing
    empty bins).

    This function computes a warp function F(freq), defined between low_freq
    and high_freq inclusive, with the following properties:
        F(low_freq) == low_freq
        F(high_freq) == high_freq
    The function is continuous and piecewise linear with two inflection
        points.
    The lower inflection point (measured in terms of the unwarped
        frequency) is at frequency l, determined as described below.
    The higher inflection point is at a frequency h, determined as
        described below.
    If l <= f <= h, then F(f) = f/vtln_warp_factor.
    If the higher inflection point (measured in terms of the unwarped
        frequency) is at h, then max(h, F(h)) == vtln_high_cutoff.
        Since (by the last point) F(h) == h/vtln_warp_factor, then
        max(h, h/vtln_warp_factor) == vtln_high_cutoff, so
        h = vtln_high_cutoff / max(1, 1/vtln_warp_factor).
          = vtln_high_cutoff * min(1, vtln_warp_factor).
    If the lower inflection point (measured in terms of the unwarped
        frequency) is at l, then min(l, F(l)) == vtln_low_cutoff
        This implies that l = vtln_low_cutoff / min(1, 1/vtln_warp_factor)
                            = vtln_low_cutoff * max(1, vtln_warp_factor)
    Args:
        vtln_low_cutoff (float): Lower frequency cutoffs for VTLN
        vtln_high_cutoff (float): Upper frequency cutoffs for VTLN
        low_freq (float): Lower frequency cutoffs in mel computation
        high_freq (float): Upper frequency cutoffs in mel computation
        vtln_warp_factor (float): Vtln warp factor
        freq (np.ndarray): given frequency in Hz

    Returns:
        np.ndarray: Freq after vtln warp
    """
    assert vtln_low_cutoff > low_freq, "be sure to set the vtln_low option higher than low_freq"
    assert vtln_high_cutoff < high_freq, "be sure to set the vtln_high option lower than high_freq [or negative]"
    l = vtln_low_cutoff * max(1.0, vtln_warp_factor)
    h = vtln_high_cutoff * min(1.0, vtln_warp_factor)
    scale = 1.0 / vtln_warp_factor
    Fl = scale * l  # F(l)
    Fh = scale * h  # F(h)
    assert l > low_freq and h < high_freq
    # slope of left part of the 3-piece linear function
    scale_left = (Fl - low_freq) / (l - low_freq)
    # [slope of center part is just "scale"]

    # slope of right part of the 3-piece linear function
    scale_right = (high_freq - Fh) / (high_freq - h)

    res = np.empty_like(freq)

    outside_low_high_freq = np.logical_or(freq < low_freq, freq > high_freq)
    before_l = freq < l
    before_h = freq < h
    after_h = freq >= h

    # order of operations matter here (since there is overlapping frequency regions)
    res[after_h] = high_freq + scale_right * (freq[after_h] - high_freq)
    res[before_h] = scale * freq[before_h]
    res[before_l] = low_freq + scale_left * (freq[before_l] - low_freq)
    res[outside_low_high_freq] = freq[outside_low_high_freq]

    return res


def vtln_warp_mel_freq(
    vtln_low_cutoff: float,
    vtln_high_cutoff: float,
    low_freq,
    high_freq: float,
    vtln_warp_factor: float,
    mel_freq: np.ndarray,
) -> np.ndarray:
    r"""
    Args:
        vtln_low_cutoff (float): Lower frequency cutoffs for VTLN
        vtln_high_cutoff (float): Upper frequency cutoffs for VTLN
        low_freq (float): Lower frequency cutoffs in mel computation
        high_freq (float): Upper frequency cutoffs in mel computation
        vtln_warp_factor (float): Vtln warp factor
        mel_freq (np.ndarray): Given frequency in Mel

    Returns:
        Tensor: ``mel_freq`` after vtln warp
    """
    return mel_scale(
        vtln_warp_freq(
            vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor, inverse_mel_scale(mel_freq)
        )
    )


def get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
    vtln_low: float,
    vtln_high: float,
    vtln_warp_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        (np.ndarray, np.ndarray): The tuple consists of ``bins`` (which is
        melbank of size (``num_bins``, ``num_fft_bins``)) and ``center_freqs`` (which is
        center frequencies of bins of size (``num_bins``)).
    """
    assert num_bins > 3, "Must have at least 3 mel bins"
    assert window_length_padded % 2 == 0
    num_fft_bins = window_length_padded / 2
    nyquist = 0.5 * sample_freq

    if high_freq <= 0.0:
        high_freq += nyquist

    assert (
        (0.0 <= low_freq < nyquist) and (0.0 < high_freq <= nyquist) and (low_freq < high_freq)
    ), "Bad values in options: low-freq {} and high-freq {} vs. nyquist {}".format(low_freq, high_freq, nyquist)

    # fft-bin width [think of it as Nyquist-freq / half-window-length]
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)

    # divide by num_bins+1 in next line because of end-effects where the bins
    # spread out to the sides.
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    if vtln_high < 0.0:
        vtln_high += nyquist

    assert vtln_warp_factor == 1.0 or (
        (low_freq < vtln_low < high_freq) and (0.0 < vtln_high < high_freq) and (vtln_low < vtln_high)
    ), "Bad values in options: vtln-low {} and vtln-high {}, versus " "low-freq {} and high-freq {}".format(
        vtln_low, vtln_high, low_freq, high_freq
    )

    bin = np.arange(num_bins).reshape((-1, 1))
    left_mel = mel_low_freq + bin * mel_freq_delta  # size(num_bins, 1)
    center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta  # size(num_bins, 1)
    right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta  # size(num_bins, 1)

    if vtln_warp_factor != 1.0:
        left_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, left_mel)
        center_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, center_mel)
        right_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, right_mel)

    center_freqs = inverse_mel_scale(center_mel)  # size (num_bins)
    # size(1, num_fft_bins)
    mel = mel_scale(fft_bin_width * np.arange(num_fft_bins)).reshape((1, -1))

    # size (num_bins, num_fft_bins)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)

    if vtln_warp_factor == 1.0:
        # left_mel < center_mel < right_mel so we can min the two slopes and clamp negative values
        bins = np.maximum(np.zeros((1,)), np.minimum(up_slope, down_slope))
    else:
        # warping can move the order of left_mel, center_mel, right_mel anywhere
        bins = np.zeros_like(up_slope)
        up_idx = (mel > left_mel) & (mel <= center_mel)  # left_mel < mel <= center_mel
        down_idx = (mel > center_mel) & (mel < right_mel)  # center_mel < mel < right_mel
        bins[up_idx] = up_slope[up_idx]
        bins[down_idx] = down_slope[down_idx]

    return bins, center_freqs


def fbank(
    waveform: np.ndarray,
    blackman_coeff: float = 0.42,
    channel: int = -1,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    high_freq: float = 0.0,
    htk_compat: bool = False,
    low_freq: float = 20.0,
    min_duration: float = 0.0,
    num_mel_bins: int = 23,
    preemphasis_coefficient: float = 0.97,
    raw_energy: bool = True,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    subtract_mean: bool = False,
    use_energy: bool = False,
    use_log_fbank: bool = True,
    use_power: bool = True,
    vtln_high: float = -500.0,
    vtln_low: float = 100.0,
    vtln_warp: float = 1.0,
    window_type: str = POVEY,
) -> np.ndarray:
    r"""Create a fbank from a raw audio signal. This matches the input/output of Kaldi's
    compute-fbank-feats.

    Args:
        waveform (Tensor): Tensor of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff (float, optional): Constant coefficient for generalized Blackman window. (Default: ``0.42``)
        channel (int, optional): Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (Default: ``-1``)
        dither (float, optional): Dithering constant (0.0 means no dither). If you turn this off, you should set
            the energy_floor option, e.g. to 1.0 or 0.1 (Default: ``0.0``)
        energy_floor (float, optional): Floor on energy (absolute, not relative) in Spectrogram computation.  Caution:
            this floor is applied to the zeroth component, representing the total signal energy.  The floor on the
            individual spectrogram elements is fixed at std::numeric_limits<float>::epsilon(). (Default: ``1.0``)
        frame_length (float, optional): Frame length in milliseconds (Default: ``25.0``)
        frame_shift (float, optional): Frame shift in milliseconds (Default: ``10.0``)
        high_freq (float, optional): High cutoff frequency for mel bins (if <= 0, offset from Nyquist)
         (Default: ``0.0``)
        htk_compat (bool, optional): If true, put energy last.  Warning: not sufficient to get HTK compatible features
         (need to change other parameters). (Default: ``False``)
        low_freq (float, optional): Low cutoff frequency for mel bins (Default: ``20.0``)
        min_duration (float, optional): Minimum duration of segments to process (in seconds). (Default: ``0.0``)
        num_mel_bins (int, optional): Number of triangular mel-frequency bins (Default: ``23``)
        preemphasis_coefficient (float, optional): Coefficient for use in signal preemphasis (Default: ``0.97``)
        raw_energy (bool, optional): If True, compute energy before preemphasis and windowing (Default: ``True``)
        remove_dc_offset (bool, optional): Subtract mean from waveform on each frame (Default: ``True``)
        round_to_power_of_two (bool, optional): If True, round window size to power of two by zero-padding input
            to FFT. (Default: ``True``)
        sample_frequency (float, optional): Waveform data sample frequency (must match the waveform file, if
            specified there) (Default: ``16000.0``)
        snip_edges (bool, optional): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends. (Default: ``True``)
        subtract_mean (bool, optional): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (Default: ``False``)
        use_energy (bool, optional): Add an extra dimension with energy to the FBANK output. (Default: ``False``)
        use_log_fbank (bool, optional):If true, produce log-filterbank, else produce linear. (Default: ``True``)
        use_power (bool, optional): If true, use power, else use magnitude. (Default: ``True``)
        vtln_high (float, optional): High inflection point in piecewise linear VTLN warping function (if
            negative, offset from high-mel-freq (Default: ``-500.0``)
        vtln_low (float, optional): Low inflection point in piecewise linear VTLN warping function (Default: ``100.0``)
        vtln_warp (float, optional): Vtln warp factor (only applicable if vtln_map not specified) (Default: ``1.0``)
        window_type (str, optional): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')
         (Default: ``'povey'``)

    Returns:
        Tensor: A fbank identical to what Kaldi would output. The shape is (m, ``num_mel_bins + use_energy``)
        where m is calculated in _get_strided
    """
    dtype = waveform.dtype

    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient
    )

    if len(waveform) < min_duration * sample_frequency:
        # signal is too short
        return np.empty((0,), dtype=dtype)

    # strided_input, size (m, padded_window_size) and signal_log_energy, size (m)
    strided_input, signal_log_energy = _get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        window_type,
        blackman_coeff,
        snip_edges,
        raw_energy,
        energy_floor,
        dither,
        remove_dc_offset,
        preemphasis_coefficient,
    )

    # size (m, padded_window_size // 2 + 1)
    spectrum = np.abs(np.fft.rfft(strided_input))
    if use_power:
        spectrum = np.power(spectrum, 2)

    # size (num_mel_bins, padded_window_size // 2)
    mel_energies, _ = get_mel_banks(
        num_mel_bins, padded_window_size, sample_frequency, low_freq, high_freq, vtln_low, vtln_high, vtln_warp
    )
    mel_energies = mel_energies.astype(dtype=dtype)

    # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
    mel_energies = np.pad(mel_energies, ((0, 0), (0, 1)), mode="constant", constant_values=0)

    # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
    mel_energies = np.matmul(spectrum, mel_energies.T)
    if use_log_fbank:
        # avoid log of zero (which should be prevented anyway by dithering)
        mel_energies = np.log(np.maximum(mel_energies, _get_epsilon(dtype)))

    # if use_energy then add it as the last column for htk_compat == true else first column
    if use_energy:
        signal_log_energy = signal_log_energy.reshape((-1, 1))  # size (m, 1)
        # returns size (m, num_mel_bins + 1)
        if htk_compat:
            mel_energies = np.concatenate((mel_energies, signal_log_energy), axis=1)
        else:
            mel_energies = np.concatenate((signal_log_energy, mel_energies), axis=1)

    mel_energies = _subtract_column_mean(mel_energies, subtract_mean)
    return mel_energies

def _create_dct(n_mfcc: int, n_mels: int, norm: Optional[str]) -> np.ndarray:
    r"""Create a DCT transformation matrix with shape (``n_mels``, ``n_mfcc``),
    normalized depending on norm.

    .. devices:: CPU

    .. properties:: TorchScript

    Args:
        n_mfcc (int): Number of mfc coefficients to retain
        n_mels (int): Number of mel filterbanks
        norm (str or None): Norm to use (either "ortho" or None)

    Returns:
        np.ndarray: The transformation matrix, to be right-multiplied to
        row-wise data of size (``n_mels``, ``n_mfcc``).
    """

    if norm is not None and norm != "ortho":
        raise ValueError('norm must be either "ortho" or None')

    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = np.arange(float(n_mels))
    k = np.arange(float(n_mfcc)).reshape((-1, 1))
    dct = np.cos(math.pi / float(n_mels) * (n + 0.5) * k)  # size (n_mfcc, n_mels)

    if norm is None:
        dct *= 2.0
    else:
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.T

def _get_dct_matrix(num_ceps: int, num_mel_bins: int) -> np.ndarray:
    # returns a dct matrix of size (num_mel_bins, num_ceps)
    # size (num_mel_bins, num_mel_bins)
    dct_matrix = _create_dct(num_mel_bins, num_mel_bins, "ortho")
    # kaldi expects the first cepstral to be weighted sum of factor sqrt(1/num_mel_bins)
    # this would be the first column in the dct_matrix for torchaudio as it expects a
    # right multiply (which would be the first column of the kaldi's dct_matrix as kaldi
    # expects a left multiply e.g. dct_matrix * vector).
    dct_matrix[:, 0] = math.sqrt(1 / float(num_mel_bins))
    dct_matrix = dct_matrix[:, :num_ceps]
    return dct_matrix


def _get_lifter_coeffs(num_ceps: int, cepstral_lifter: float) -> np.ndarray:
    # returns size (num_ceps)
    # Compute liftering coefficients (scaling on cepstral coeffs)
    # coeffs are numbered slightly differently from HTK: the zeroth index is C0, which is not affected.
    i = np.arange(num_ceps)
    return 1.0 + 0.5 * cepstral_lifter * np.sin(math.pi * i / cepstral_lifter)


def mfcc(
    waveform: np.ndarray,
    blackman_coeff: float = 0.42,
    cepstral_lifter: float = 22.0,
    channel: int = -1,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    high_freq: float = 0.0,
    htk_compat: bool = False,
    low_freq: float = 20.0,
    num_ceps: int = 13,
    min_duration: float = 0.0,
    num_mel_bins: int = 23,
    preemphasis_coefficient: float = 0.97,
    raw_energy: bool = True,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    subtract_mean: bool = False,
    use_energy: bool = False,
    vtln_high: float = -500.0,
    vtln_low: float = 100.0,
    vtln_warp: float = 1.0,
    window_type: str = POVEY,
) -> np.ndarray:
    r"""Create a mfcc from a raw audio signal. This matches the input/output of Kaldi's
    compute-mfcc-feats.

    Args:
        waveform (np.ndaray): Tensor of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff (float, optional): Constant coefficient for generalized Blackman window. (Default: ``0.42``)
        cepstral_lifter (float, optional): Constant that controls scaling of MFCCs (Default: ``22.0``)
        channel (int, optional): Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (Default: ``-1``)
        dither (float, optional): Dithering constant (0.0 means no dither). If you turn this off, you should set
            the energy_floor option, e.g. to 1.0 or 0.1 (Default: ``0.0``)
        energy_floor (float, optional): Floor on energy (absolute, not relative) in Spectrogram computation.  Caution:
            this floor is applied to the zeroth component, representing the total signal energy.  The floor on the
            individual spectrogram elements is fixed at std::numeric_limits<float>::epsilon(). (Default: ``1.0``)
        frame_length (float, optional): Frame length in milliseconds (Default: ``25.0``)
        frame_shift (float, optional): Frame shift in milliseconds (Default: ``10.0``)
        high_freq (float, optional): High cutoff frequency for mel bins (if <= 0, offset from Nyquist)
         (Default: ``0.0``)
        htk_compat (bool, optional): If true, put energy last.  Warning: not sufficient to get HTK compatible
         features (need to change other parameters). (Default: ``False``)
        low_freq (float, optional): Low cutoff frequency for mel bins (Default: ``20.0``)
        num_ceps (int, optional): Number of cepstra in MFCC computation (including C0) (Default: ``13``)
        min_duration (float, optional): Minimum duration of segments to process (in seconds). (Default: ``0.0``)
        num_mel_bins (int, optional): Number of triangular mel-frequency bins (Default: ``23``)
        preemphasis_coefficient (float, optional): Coefficient for use in signal preemphasis (Default: ``0.97``)
        raw_energy (bool, optional): If True, compute energy before preemphasis and windowing (Default: ``True``)
        remove_dc_offset (bool, optional): Subtract mean from waveform on each frame (Default: ``True``)
        round_to_power_of_two (bool, optional): If True, round window size to power of two by zero-padding input
            to FFT. (Default: ``True``)
        sample_frequency (float, optional): Waveform data sample frequency (must match the waveform file, if
            specified there) (Default: ``16000.0``)
        snip_edges (bool, optional): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends. (Default: ``True``)
        subtract_mean (bool, optional): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (Default: ``False``)
        use_energy (bool, optional): Add an extra dimension with energy to the FBANK output. (Default: ``False``)
        vtln_high (float, optional): High inflection point in piecewise linear VTLN warping function (if
            negative, offset from high-mel-freq (Default: ``-500.0``)
        vtln_low (float, optional): Low inflection point in piecewise linear VTLN warping function (Default: ``100.0``)
        vtln_warp (float, optional): Vtln warp factor (only applicable if vtln_map not specified) (Default: ``1.0``)
        window_type (str, optional): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')
         (Default: ``"povey"``)

    Returns:
        np.ndarray: A mfcc identical to what Kaldi would output. The shape is (m, ``num_ceps``)
        where m is calculated in _get_strided
    """
    assert num_ceps <= num_mel_bins, "num_ceps cannot be larger than num_mel_bins: %d vs %d" % (num_ceps, num_mel_bins)

    dtype = waveform.dtype

    # The mel_energies should not be squared (use_power=True), not have mean subtracted
    # (subtract_mean=False), and use log (use_log_fbank=True).
    # size (m, num_mel_bins + use_energy)
    feature = fbank(
        waveform=waveform,
        blackman_coeff=blackman_coeff,
        channel=channel,
        dither=dither,
        energy_floor=energy_floor,
        frame_length=frame_length,
        frame_shift=frame_shift,
        high_freq=high_freq,
        htk_compat=htk_compat,
        low_freq=low_freq,
        min_duration=min_duration,
        num_mel_bins=num_mel_bins,
        preemphasis_coefficient=preemphasis_coefficient,
        raw_energy=raw_energy,
        remove_dc_offset=remove_dc_offset,
        round_to_power_of_two=round_to_power_of_two,
        sample_frequency=sample_frequency,
        snip_edges=snip_edges,
        subtract_mean=False,
        use_energy=use_energy,
        use_log_fbank=True,
        use_power=True,
        vtln_high=vtln_high,
        vtln_low=vtln_low,
        vtln_warp=vtln_warp,
        window_type=window_type,
    )

    if use_energy:
        # size (m)
        signal_log_energy = feature[:, num_mel_bins if htk_compat else 0]
        # offset is 0 if htk_compat==True else 1
        mel_offset = int(not htk_compat)
        feature = feature[:, mel_offset : (num_mel_bins + mel_offset)]

    # size (num_mel_bins, num_ceps)
    dct_matrix = _get_dct_matrix(num_ceps, num_mel_bins).astype(dtype)

    # size (m, num_ceps)
    feature = np.matmul(feature, dct_matrix)

    if cepstral_lifter != 0.0:
        # size (1, num_ceps)
        lifter_coeffs = _get_lifter_coeffs(num_ceps, cepstral_lifter).reshape((1, -1))
        feature *= lifter_coeffs.astype(dtype)

    # if use_energy then replace the last column for htk_compat == true else first column
    if use_energy:
        feature[:, 0] = signal_log_energy

    if htk_compat:
        energy = feature[:, 0].reshape((-1, 1))  # size (m, 1)
        feature = feature[:, 1:]  # size (m, num_ceps - 1)
        if not use_energy:
            # scale on C0 (actually removing a scale we previously added that's
            # part of one common definition of the cosine transform.)
            energy *= math.sqrt(2)

        feature = np.concatenate((feature, energy), axis=1)

    feature = _subtract_column_mean(feature, subtract_mean)
    return feature
