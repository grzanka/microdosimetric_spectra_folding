import pytest
import numpy as np
from src.spectrum import first_moment

@pytest.fixture
def example_data():
    bin_centers = np.array([1, 2, 3, 4, 5])
    bin_values = np.array([10, 20, 30, 40, 50])
    return bin_centers, bin_values

def test_first_moment(example_data):
    bin_centers, bin_values = example_data
    result = first_moment(bin_centers, bin_values)
    expected_result = np.sum(bin_centers * bin_values) / np.sum(bin_values)
    assert result == expected_result

def test_first_moment_with_zeros():
    bin_centers = np.array([1, 2, 3, 4, 5])
    bin_values = np.array([0, 0, 0, 0, 0])
    with pytest.raises(ZeroDivisionError):
        first_moment(bin_centers, bin_values)

def test_moment_olko_phd_thesis_fig3p3(spectrum_fig3p3_olko_phd):
    result = first_moment(spectrum_fig3p3_olko_phd.bin_centers, spectrum_fig3p3_olko_phd.bin_values_fy)
    expected_result = 2
    assert result == pytest.approx(expected_result)
    # check normalized spectrum as well
    result_from_normalized = first_moment(spectrum_fig3p3_olko_phd.bin_centers, spectrum_fig3p3_olko_phd.bin_values_fy_normalized)
    assert result_from_normalized == pytest.approx(expected_result)


def test_yF(spectrum_fig3p3_olko_phd):
    assert spectrum_fig3p3_olko_phd.yF == pytest.approx(2.0)
