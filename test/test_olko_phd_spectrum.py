import numpy as np
import pytest

from src.spectrum import SpectrumData

def test_y(spectrum_data_fig3p3_olko_phd: SpectrumData):
    assert np.array_equal(spectrum_data_fig3p3_olko_phd.bin_centers, [1,2,3])
    assert spectrum_data_fig3p3_olko_phd.bin_centers.shape == (3,)
    assert spectrum_data_fig3p3_olko_phd.bin_centers.ndim == 1


def test_fy(spectrum_data_fig3p3_olko_phd: SpectrumData):
    assert np.array_equal(spectrum_data_fig3p3_olko_phd.bin_values_freq, [2,2,2])
    assert np.unique(spectrum_data_fig3p3_olko_phd.bin_values_freq) == pytest.approx(2)
    assert spectrum_data_fig3p3_olko_phd.bin_values_freq.shape == (3,)
    assert spectrum_data_fig3p3_olko_phd.bin_values_freq.ndim == 1
    assert spectrum_data_fig3p3_olko_phd.norm == 6
    assert np.unique(spectrum_data_fig3p3_olko_phd.bin_values_freq_normalized) == pytest.approx(1/3)

def test_yfy(spectrum_data_fig3p3_olko_phd: SpectrumData):
    assert np.array_equal(spectrum_data_fig3p3_olko_phd.bin_values_freq_times_x, [2,4,6])
    assert np.array_equal(spectrum_data_fig3p3_olko_phd.bin_values_freq_times_x_normalized, [1/3,2/3,1])

def test_dy(spectrum_data_fig3p3_olko_phd: SpectrumData):
    assert np.array_equal(spectrum_data_fig3p3_olko_phd.bin_values_dose, [1,2,3])

def test_dy(spectrum_data_fig3p3_olko_phd: SpectrumData):
    assert np.array_equal(spectrum_data_fig3p3_olko_phd.bin_values_dose_times_x, [1,4,9])
