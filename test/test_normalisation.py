import numpy as np
import pytest
from src.spectrum import Spectrum, SpectrumBinningType, from_str

def test_sum_of_f(small_spectrum: Spectrum):
    assert small_spectrum.bin_values_fy.sum() == pytest.approx(1.0), "Sum of f is not 1.0"
    assert small_spectrum.fy.sum() == pytest.approx(1.0), "Sum of fy is not 1.0"
    assert small_spectrum.f_sum == pytest.approx(1.0), "f_sum is not 1.0"
    assert small_spectrum.norm == pytest.approx(1.0), "norm is not 1.0"

def test_normalized_f(not_normalised_spectrum: Spectrum):
    assert not_normalised_spectrum.bin_values_fy.sum() == pytest.approx(10.0), "Sum of f is not 10.0"
    assert not_normalised_spectrum.f_sum == pytest.approx(10.0), "f_sum is not 10.0"
    assert not_normalised_spectrum.bin_values_fy_normalized.sum() == pytest.approx(1.0), "Sum of normalized f is not 1.0"
    assert not_normalised_spectrum.norm == pytest.approx(10.0), "norm is not 10.0"