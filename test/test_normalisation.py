import pytest
from src.spectrum import Spectrum

def test_sum_of_f(small_spectrum: Spectrum):
    assert small_spectrum.bin_values_fy.sum() == pytest.approx(1.0), "Sum of f is not 1.0"
    assert small_spectrum.fy.sum() == pytest.approx(1.0), "Sum of fy is not 1.0"
    assert small_spectrum.norm == pytest.approx(1.0), "norm is not 1.0"

def test_normalized_f(not_normalised_spectrum: Spectrum):
    assert not_normalised_spectrum.bin_values_fy.sum() == pytest.approx(10.0), "Sum of f is not 10.0"
    assert not_normalised_spectrum.bin_values_fy_normalized.sum() == pytest.approx(1.0), "Sum of normalized f is not 1.0"
    assert not_normalised_spectrum.norm == pytest.approx(10.0), "norm is not 10.0"

def test_norm_unknown_binning(spectrum_unknown_binning : Spectrum):    
    assert spectrum_unknown_binning.norm == pytest.approx(1.2), "norm is not 1.0"