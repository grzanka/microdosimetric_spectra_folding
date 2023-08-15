import pytest
from src.spectrum import Spectrum

@pytest.fixture
def spectrum_fig3p3_olko_phd() -> Spectrum:
    bin_centers = [1,2,3]
    bin_values = [2,2,2]
    return Spectrum.from_lists(bin_centers_list=bin_centers, bin_values_list=bin_values)