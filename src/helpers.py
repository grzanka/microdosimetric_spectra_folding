from enum import Enum, auto
import logging
import numpy as np
from numpy.typing import NDArray

class SpectrumValueType(Enum):
    '''Enum class for spectrum value types.'''
    freq = auto()
    freq_times_bin_centers = auto()
    dose_times_bin_centers = auto()
    fy = freq
    yfy = freq_times_bin_centers
    ydy = dose_times_bin_centers
    fz = freq
    zfz = freq_times_bin_centers
    zdz = dose_times_bin_centers

class SpectrumBinningType(Enum):
    '''Enum class for spectrum binning types.'''
    log = auto()
    linear = auto()
    unknown = auto()
    lin = linear

def first_moment(bin_edges: NDArray, bin_values: NDArray) -> float:
    '''Calculate the first moment of a spectrum. It may be not normalized.'''
    if bin_values.sum() == 0:
        return np.nan
    bin_widths = np.diff(bin_edges)
    nominator = (0.5 * (bin_edges[1:]**2-bin_edges[:-1]**2) * bin_values).sum()
    denominator = (bin_values * bin_widths).sum()
    return nominator / denominator


def binning_type(bin_centers : NDArray) -> SpectrumBinningType:
    '''Determine the binning type from bin_centers.'''
    result = SpectrumBinningType.unknown
    # check if bin_centers form an arithmetic progression
    if bin_centers.size >= 2 and np.allclose(np.diff(bin_centers), bin_centers[1] - bin_centers[0]):
        result = SpectrumBinningType.linear
    # check if bin_centers form a geometric progression
    elif bin_centers.size >= 2 and np.allclose(np.diff(np.log(bin_centers)), np.log(bin_centers[1]) - np.log(bin_centers[0])):
        result = SpectrumBinningType.log
    return result

def bin_edges(bin_centers : NDArray, binning_type: SpectrumBinningType) -> NDArray:
    '''Calculate bin edges from bin centers.'''
    result = np.empty(0)
    if binning_type == SpectrumBinningType.linear:
        bin_centers_diff = np.diff(bin_centers).mean()
        logging.debug("bin_centers_diff is {}".format(bin_centers_diff))
        result = np.append(bin_centers - bin_centers_diff / 2, bin_centers[-1] + bin_centers_diff / 2)
    if binning_type == SpectrumBinningType.log:
        bin_centers_ratio = np.exp(np.diff(np.log(bin_centers)).mean())
        logging.debug("bin_centers_ratio is {}".format(bin_centers_ratio))
        result = np.append(bin_centers / np.sqrt(bin_centers_ratio), bin_centers[-1] * np.sqrt(bin_centers_ratio))
    if binning_type == SpectrumBinningType.unknown and bin_centers.size > 1:
        bin_centers_diff = np.diff(bin_centers)
        lowest_bin_edge = bin_centers[0] - bin_centers_diff[0] / 2
        highest_bin_edge = bin_centers[-1] + bin_centers_diff[-1] / 2
        middle_bin_edges = bin_centers[:-1] + bin_centers_diff / 2
        result = np.append(np.append(lowest_bin_edge, middle_bin_edges), highest_bin_edge)
    if bin_centers.size == 1:
        result = np.array([np.nan, np.nan])
    return result


def others_from_freq_arrays(x: NDArray, freq: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    '''Calculate yfy/zfz and ydy/zdz from y/z and freq (fy/fz).'''

    xfx = x * freq # yfy = y * f(y)
    edges = bin_edges(bin_centers=x, binning_type=binning_type(bin_centers=x))
    xF = first_moment(bin_edges=edges, bin_values=freq)

    # d(x) = (x / xF) * f(x)
    dx = (x / xF) * freq
    xdx = x * dx

    return xfx, dx, xdx

def others_from_x_times_freq(x: NDArray, x_times_freq: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    '''Calculate fy/fz (freq), dy/dz (dose) and ydy/zdz (dose times x) from y/z and yfy/zfz.'''

    freq = x_times_freq / x # yfy = y * f(y) / zfz = z * f(z)
    edges = bin_edges(bin_centers=x, binning_type=binning_type(bin_centers=x))
    xF = first_moment(bin_edges=edges, bin_values=freq)

    # d(y) = (y / yF) * f(y)
    dose = (x / xF) * freq
    dose_times_x = x * dose
    return freq, dose, dose_times_x

def others_from_dose_arrays(x: NDArray, dose: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    '''Calculate fy/fz, yfy/zfz and ydy/zdz from y/z and dose (dy/dz).'''

    xdx = x * dose # ydy = y * d(y) / zdz = z * d(z)
    edges = bin_edges(bin_centers=x, binning_type=binning_type(bin_centers=x))
    widths = np.diff(edges)

    # d(x) = (x / xF) * f(x)
    # f_unnorm(x) = d(x) / x
    # calculate unnormalized freq distribution as we cannot determine xF yet
    freq_not_norm = dose / x
    norm = widths @ freq_not_norm
    freq = freq_not_norm / norm if norm != 0 else np.zeros_like(freq_not_norm)

    x_times_freq = x * freq

    return freq, x_times_freq, xdx

def others_from_x_times_dose_arrays(x: NDArray, x_times_dose: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    '''Calculate fy/fz, yfy/zfz and dy/dz from y/z and ydy/zdz.'''

    dose = x_times_dose / x # ydy = y * d(y) / zdz = z * d(z)
    edges = bin_edges(bin_centers=x, binning_type=binning_type(bin_centers=x))
    widths = np.diff(edges)

    # d(x) = (x / xF) * f(x)
    # f_unnorm(x) = d(x) / x
    # calculate unnormalized freq distribution as we cannot determine xF yet
    freq_not_norm = dose / x
    norm = widths @ freq_not_norm
    freq = freq_not_norm / norm if norm != 0 else np.zeros_like(freq_not_norm)

    x_times_freq = x * freq

    return freq, x_times_freq, dose


def normalized_fy(y: NDArray, fy: NDArray, norm: float) -> NDArray:
    '''Calculate normalized fy from y and fy.'''
    fy_normalized = fy / norm
    yfy, _, _ = others_from_freq_arrays(x=y, freq=fy)
    return yfy / yfy.sum()
