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
