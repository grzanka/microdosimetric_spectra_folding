import logging
import numpy as np
import pytest
from src.spectrum import Spectrum, SpectrumBinningType, from_str



def test_bin_centers(spectrum_log_binning: Spectrum):
    assert spectrum_log_binning.bin_centers.shape == (4,)
    assert spectrum_log_binning.bin_centers.ndim == 1
    assert spectrum_log_binning.binning_type == SpectrumBinningType.log

def test_bin_centers(small_spectrum: Spectrum):
    assert np.allclose(small_spectrum.bin_centers, np.array([1, 2, 3, 4]))
    assert small_spectrum.bin_centers.shape == (4,)
    assert small_spectrum.bin_centers.ndim == 1
    assert small_spectrum.binning_type == SpectrumBinningType.linear

def test_bin_edges_linear(small_spectrum: Spectrum):
    assert np.allclose(small_spectrum.bin_edges, np.array([0.5, 1.5, 2.5, 3.5, 4.5]))
    assert small_spectrum.bin_edges.shape == (5,)
    assert small_spectrum.bin_edges.ndim == 1
    assert small_spectrum.binning_type == SpectrumBinningType.linear

def test_bins_edges_log(spectrum_log_binning: Spectrum):
    expected_bin_edges = np.array([0.1 / np.sqrt(10), 1 / np.sqrt(10), 10 / np.sqrt(10), 100 / np.sqrt(10), 100 * np.sqrt(10)])
    assert np.allclose(spectrum_log_binning.bin_edges, expected_bin_edges)
    assert spectrum_log_binning.bin_edges.shape == (5,)
    assert spectrum_log_binning.bin_edges.ndim == 1
    assert spectrum_log_binning.binning_type == SpectrumBinningType.log

def test_bin_widths_linear(small_spectrum: Spectrum):
    assert np.allclose(small_spectrum.bin_widths, np.array([1, 1, 1, 1]))

def test_bin_widths_log(spectrum_log_binning: Spectrum):
    expected_bin_widths = spectrum_log_binning.bin_centers * (np.sqrt(10) - np.sqrt(1/10))
    assert np.allclose(spectrum_log_binning.bin_widths, expected_bin_widths)

def test_uknown_binning(spectrum_unknown_binning: Spectrum):
    assert spectrum_unknown_binning.bin_edges.size == 5
    assert np.array_equal(spectrum_unknown_binning.bin_edges, np.array([0.5, 1.5, 3, 4.5, 5.5]))
    assert spectrum_unknown_binning.bin_widths.size == 4
    assert np.array_equal(spectrum_unknown_binning.bin_widths, np.array([1, 1.5, 1.5, 1]))
    assert spectrum_unknown_binning.binning_type == SpectrumBinningType.unknown

def test_bin_numbers(small_spectrum: Spectrum):
    # bin centers 1 2 3 4
    # bin edges 0.5 1.5 2.5 3.5 4.5
    # bins:
    # as -1: <-inf, 0.5)
    # 0: [0.5, 1.5)
    # 1: [1.5, 2.5)
    # 2: [2.5, 3.5)
    # 3: [3.5, 4.5)
    # as 4: [4.5, inf)
    y_values = np.array([
        -1, 0, 0.3,  # bin -1
        0.5, 1, 1.2, # bin 0
        1.5, 2.3,  # bin 1
        2.5, 3, # bin 2
        3.5, 4, # bin 3
        4.5, 5 # bin 4
        ]) 
    expected_bin_numbers = np.array([
        -1, -1, -1, 
        0, 0, 0,
        1, 1,
        2, 2,
        3, 3,
        4, 4
        ])
    assert np.array_equal(small_spectrum.bin_numbers(y=y_values), expected_bin_numbers)
    with pytest.raises(TypeError):
        small_spectrum.bin_centers(y=2)
    assert np.array_equal(small_spectrum.bin_numbers(y=[2]), [1])

def test_bin_numbers_log_spectrum(spectrum_log_binning: Spectrum):
    # bin centers 0.1, 1, 10, 100
    # bin edges 0.03162278  0.31622777  3.16227766 31.6227766  316.227766
    # bins:
    # as -1: <-inf, 0.03162278)
    # 0: [0.03162278, 0.31622777)
    # 1: [0.31622777, 3.16227766)
    # 2: [3.16227766, 31.6227766)
    # 3: [31.6227766, 316.227766)
    # as 4: [316.227766, inf)
    y_values = np.array([
        -1, 0, 0.03,  # bin -1
        0.05, 0.3, # bin 0
        0.5, 1, 3,  # bin 1
        5, 10, 30, # bin 2
        32, 100, # bin 3
        317, 1000 # bin 4
        ]) 
    expected_bin_numbers = np.array([
        -1, -1, -1, 
        0, 0,
        1, 1, 1,
        2, 2, 2,
        3, 3,
        4, 4
        ])
    assert np.array_equal(spectrum_log_binning.bin_numbers(y=y_values), expected_bin_numbers)
    with pytest.raises(TypeError):
        spectrum_log_binning.bin_centers(y=2)
    assert np.array_equal(spectrum_log_binning.bin_numbers(y=[2]), [1])


def test_bin_numbers_unknown_binning(spectrum_unknown_binning: Spectrum):
    # bin centers 1 2 4 5
    # bin edges 0.5 1.5 3 4.5 5.5
    # bins:
    # as -1: <-inf, 0.5)
    # 0: [0.5, 1.5)
    # 1: [1.5, 3)
    # 2: [3, 4.5)
    # 3: [4.5, 5.5)
    # as 4: [5.5, inf)
    y_values = np.array([
        -1, 0, 0.3,  # bin -1
        0.5, 1, 1.2, # bin 0
        1.5, 2.3,  # bin 1
        3, 3.5, 4, # bin 2
        4.5, 5, # bin 3
        5.5, 6 # bin 4
        ]) 
    expected_bin_numbers = np.array([
        -1, -1, -1, 
        0, 0, 0,
        1, 1,
        2, 2, 2,
        3, 3,
        4, 4
        ])
    assert np.array_equal(spectrum_unknown_binning.bin_numbers(y=[4]), [2])
    logging.debug(f"bin numbers {spectrum_unknown_binning.bin_numbers(y=y_values)}")
    assert np.array_equal(spectrum_unknown_binning.bin_numbers(y=y_values), expected_bin_numbers)
    with pytest.raises(TypeError):
        spectrum_unknown_binning.bin_centers(y=4)
