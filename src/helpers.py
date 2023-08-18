from enum import Enum, auto
import logging
import numpy as np
from numpy.typing import NDArray

class SpectrumValueType(Enum):
    '''Enum class for spectrum value types.'''
    fy = auto()
    yfy = auto()
    ydy = auto()

class SpectrumBinningType(Enum):
    '''Enum class for spectrum binning types.'''
    log = auto()
    linear = auto()
    unknown = auto()

def first_moment(bin_centers: NDArray, bin_values: NDArray) -> float:
    '''Calculate the first moment of a spectrum. It may be not normalized.'''
    if bin_values.sum() == 0:
        raise ZeroDivisionError("Sum of bin_values must be positive")
    return np.sum(bin_centers * bin_values) / np.sum(bin_values)

def binning_type(bin_centers : NDArray) -> SpectrumBinningType:
    '''Determine the binning type from bin_centers.'''
    result = SpectrumBinningType.unknown
    # check if bin_centers form an arithmetic progression
    if bin_centers.size >= 2 and np.all(np.diff(bin_centers) == bin_centers[1] - bin_centers[0]):
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
