import numpy as np
import pytest
from src.spectrum import Spectrum, SpectrumBinningType

@pytest.fixture
def spectrum_log_binning() -> Spectrum:
    bin_centers = [0.1, 1, 10, 100]
    bin_values_fy = [0.1, 0.2, 0.3, 0.4]
    return Spectrum.from_lists(bin_centers_list=bin_centers, bin_values_list=bin_values_fy)

def test_bin_centers(spectrum_log_binning: Spectrum):
    assert spectrum_log_binning.bin_centers.shape == (4,)
    assert spectrum_log_binning.bin_centers.ndim == 1
    assert spectrum_log_binning.binning_type == SpectrumBinningType.log