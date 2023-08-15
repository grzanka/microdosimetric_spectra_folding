import numpy as np
import pytest
from src.spectrum import Spectrum


@pytest.fixture
def small_spectrum() -> Spectrum:
    bin_centers = [1, 2, 3, 4]
    bin_values_f = [0.1, 0.2, 0.3, 0.4]
    return Spectrum.from_lists(bin_centers_list=bin_centers, bin_values_list=bin_values_f)


@pytest.fixture
def not_normalised_spectrum() -> Spectrum:
    bin_centers = [1, 2, 3, 4]
    bin_values_f = [1, 2, 3, 4]
    return Spectrum.from_lists(bin_centers_list=bin_centers, bin_values_list=bin_values_f)

def test_bin_centers(small_spectrum: Spectrum):
    assert np.array_equal(small_spectrum.bin_centers, np.array([1, 2, 3, 4]))

def test_sum_of_f(small_spectrum: Spectrum):
    assert small_spectrum.bin_values_f.sum() == pytest.approx(1.0)
    assert small_spectrum.fy.sum() == pytest.approx(1.0)
    assert small_spectrum.f_sum == pytest.approx(1.0)

def test_normalized_f(not_normalised_spectrum: Spectrum):
    assert not_normalised_spectrum.bin_values_f.sum() == pytest.approx(10.0)
    assert not_normalised_spectrum.f_sum == pytest.approx(10.0)
    assert not_normalised_spectrum.bin_values_f_normalized.sum() == pytest.approx(1.0)

def test_creation_from_lists(spectrum_fig3p3_olko_phd):
    bin_centers = spectrum_fig3p3_olko_phd.bin_centers.tolist()
    bin_values_f = spectrum_fig3p3_olko_phd.bin_values_f.tolist()

    spectrum = Spectrum.from_lists(bin_centers, bin_values_list=bin_values_f)
    assert spectrum.num_bins == len(bin_centers)
    assert np.array_equal(spectrum.bin_values_f, spectrum_fig3p3_olko_phd.bin_values_f)
    assert np.array_equal(spectrum.yF, spectrum_fig3p3_olko_phd.yF)
    assert np.array_equal(spectrum.fy, spectrum_fig3p3_olko_phd.fy)
    assert np.array_equal(spectrum.yfy, spectrum_fig3p3_olko_phd.yfy)
    assert np.array_equal(spectrum.ydy, spectrum_fig3p3_olko_phd.ydy)

def test_invalid_initialization():
    with pytest.raises(ValueError):
        Spectrum()

    bin_centers = [1, 2, 3, 4, 5]
    bin_values_f = [3, 6, 2, 8, 5]
    bin_values_yfy = [2, 4, 6, 8, 10]

    # Trying to initialize with conflicting values
    with pytest.raises(ValueError):
        Spectrum.from_lists(bin_centers, bin_values_list=bin_values_f, bin_values_yfy_list=bin_values_yfy)

    # Trying to initialize with missing values
    with pytest.raises(ValueError):
        Spectrum.from_lists(bin_centers)
