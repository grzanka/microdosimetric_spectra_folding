import logging
import pytest
import numpy as np
from src.spectrum import SpectrumData, first_moment


@pytest.fixture
def example_data():
    bin_centers = np.array([1, 2, 3, 4, 5])
    bin_values = np.array([10, 20, 30, 40, 50])
    return bin_centers, bin_values


def test_first_moment(example_data):
    bin_centers, bin_values = example_data
    logging.debug(f"bin_centers: {bin_centers}")
    logging.debug(f"bin_values: {bin_values}")
    example_spectrum = SpectrumData(bin_centers=bin_centers, bin_values_freq=bin_values)
    result = first_moment(example_spectrum.bin_edges, bin_values)
    expected_result = np.sum(bin_centers * bin_values) / np.sum(bin_values)
    assert result == pytest.approx(expected_result)
    assert result == pytest.approx(example_spectrum.freq_mean)


def test_first_moment_with_zeros():
    bin_centers = np.array([1, 2, 3, 4, 5])
    bin_values = np.array([0, 0, 0, 0, 0])
    example_spectrum = SpectrumData(bin_centers=bin_centers, bin_values_freq=bin_values)
    assert np.isnan(
        first_moment(example_spectrum.bin_edges, bin_values)
    ), "First moment of zero values must be NaN"


def test_moment_olko_phd_thesis_fig3p3(spectrum_data_fig3p3_olko_phd: SpectrumData):
    result = first_moment(
        spectrum_data_fig3p3_olko_phd.bin_edges,
        spectrum_data_fig3p3_olko_phd.bin_values_freq,
    )
    expected_result = 2
    assert result == pytest.approx(expected_result)
    # check normalized spectrum as well
    result_from_normalized = first_moment(
        spectrum_data_fig3p3_olko_phd.bin_edges,
        spectrum_data_fig3p3_olko_phd.bin_values_freq_normalized,
    )
    assert result_from_normalized == pytest.approx(expected_result)


def test_freq_mean(spectrum_data_fig3p3_olko_phd: SpectrumData):
    assert spectrum_data_fig3p3_olko_phd.freq_mean == pytest.approx(2.0)
