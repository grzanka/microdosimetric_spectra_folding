import numpy as np
import pytest
from src.spectrum import Spectrum, from_str


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

def test_empty_spectrum():
    empty_spectrum = Spectrum()
    assert empty_spectrum.num_bins == 0

def test_bin_centers(small_spectrum: Spectrum):
    assert np.array_equal(small_spectrum.bin_centers, np.array([1, 2, 3, 4]))

def test_sum_of_f(small_spectrum: Spectrum):
    assert small_spectrum.bin_values_fy.sum() == pytest.approx(1.0)
    assert small_spectrum.fy.sum() == pytest.approx(1.0)
    assert small_spectrum.f_sum == pytest.approx(1.0)

def test_normalized_f(not_normalised_spectrum: Spectrum):
    assert not_normalised_spectrum.bin_values_fy.sum() == pytest.approx(10.0)
    assert not_normalised_spectrum.f_sum == pytest.approx(10.0)
    assert not_normalised_spectrum.bin_values_fy_normalized.sum() == pytest.approx(1.0)

def test_creation_from_lists(spectrum_fig3p3_olko_phd):
    bin_centers = spectrum_fig3p3_olko_phd.bin_centers.tolist()
    bin_values_fy = spectrum_fig3p3_olko_phd.bin_values_fy.tolist()

    spectrum = Spectrum.from_lists(bin_centers, bin_values_list=bin_values_fy)
    assert spectrum.num_bins == len(bin_centers)
    assert np.array_equal(spectrum.bin_values_fy, spectrum_fig3p3_olko_phd.bin_values_fy)
    assert np.array_equal(spectrum.yF, spectrum_fig3p3_olko_phd.yF)
    assert np.array_equal(spectrum.fy, spectrum_fig3p3_olko_phd.fy)
    assert np.array_equal(spectrum.yfy, spectrum_fig3p3_olko_phd.yfy)
    assert np.array_equal(spectrum.ydy, spectrum_fig3p3_olko_phd.ydy)

def test_invalid_initialization():
    bin_centers = [1, 2, 3, 4, 5]
    bin_values_fy = [3, 6, 2, 8, 5]
    bin_values_yfy = [2, 4, 6, 8, 10]

    # Trying to initialize with conflicting values
    with pytest.raises(ValueError):
        Spectrum.from_lists(bin_centers, bin_values_list=bin_values_fy, bin_values_yfy_list=bin_values_yfy)

    # Trying to initialize with missing values
    with pytest.raises(ValueError):
        Spectrum.from_lists(bin_centers)


def test_if_printout_has_multiple_lines(small_spectrum: Spectrum, capsys):
    print(small_spectrum)
    captured = capsys.readouterr()
    output_lines = captured.out.splitlines()
    assert len(output_lines) > 1

def test_loading_from_str_with_fy():
    empty_str = ""
    with pytest.raises(ValueError):
        from_str(empty_str)
    corrupts_str = "1 2 3 4 5 6 7 8 9 10"
    with pytest.raises(ValueError):
        from_str(corrupts_str)
    two_rows_str = "1 2 3 4 5 6 7 8 9 10\n1 2 3 4 5 6 7 8 9 10"
    with pytest.raises(ValueError):
        from_str(two_rows_str)
    two_colums_str = "1 2\n3 4\n5 6\n7 8\n9 10"
    spectrum = from_str(two_colums_str)
    assert spectrum.num_bins == 5
    str_with_commas = "1,2\n3,4\n5,6\n7,8\n9,10"
    spectrum = from_str(str_with_commas, delimiter=",")
    assert spectrum.num_bins == 5