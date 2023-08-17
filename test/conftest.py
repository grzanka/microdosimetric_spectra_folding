import pytest
from src.spectrum import Spectrum

@pytest.fixture
def spectrum_fig3p3_olko_phd() -> Spectrum:
    bin_centers = [1,2,3]
    bin_values = [2,2,2]
    return Spectrum.from_lists(bin_centers_list=bin_centers, bin_values_list=bin_values)

@pytest.fixture
def small_spectrum() -> Spectrum:
    bin_centers = [1, 2, 3, 4]
    bin_values_fy = [0.1, 0.2, 0.3, 0.4]
    return Spectrum.from_lists(bin_centers_list=bin_centers, bin_values_list=bin_values_fy)

@pytest.fixture
def not_normalised_spectrum() -> Spectrum:
    bin_centers = [1, 2, 3, 4]
    bin_values_fy = [1, 2, 3, 4]
    return Spectrum.from_lists(bin_centers_list=bin_centers, bin_values_list=bin_values_fy)

@pytest.fixture
def spectrum_log_binning() -> Spectrum:
    bin_centers = [0.1, 1, 10, 100]
    bin_values_fy = [0.1, 0.2, 0.3, 0.4]
    return Spectrum.from_lists(bin_centers_list=bin_centers, bin_values_list=bin_values_fy)