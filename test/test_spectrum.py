import numpy as np
import pytest
from src.spectrum import Spectrum, SpectrumBinningType, from_str


def test_empty_spectrum():
    empty_spectrum = Spectrum()
    assert empty_spectrum.num_bins == 0
    assert empty_spectrum.binning_type == SpectrumBinningType.unknown

def test_negative_spectrum():
    with pytest.raises(ValueError):
        Spectrum.from_lists([-1, 0, 1], [-1, -2, -3])

def test_spectrum_with_one_bin():
    spectrum = Spectrum.from_lists([1], [1])
    assert spectrum.num_bins == 1
    assert spectrum.binning_type == SpectrumBinningType.unknown
    assert spectrum.bin_centers.size == 1
    assert spectrum.bin_centers[0] == pytest.approx(1)
    assert spectrum.bin_widths.size == 1
    assert np.isnan(spectrum.bin_widths[0])
    assert spectrum.bin_edges.size == 2
    assert np.isnan(spectrum.bin_edges[0])
    assert np.isnan(spectrum.bin_edges[1])

def test_spectrum_with_zero_bin_values():
    spectrum = Spectrum.from_lists([1, 2, 3], [0, 0, 0])
    assert spectrum.num_bins == 3
    assert spectrum.binning_type == SpectrumBinningType.linear
    assert np.isnan(spectrum.yF), "yF must be NaN if bin_values are all zero"



def test_creation_from_lists(spectrum_fig3p3_olko_phd):
    bin_centers = spectrum_fig3p3_olko_phd.bin_centers.tolist()
    bin_values_fy = spectrum_fig3p3_olko_phd.bin_values_fy.tolist()

    spectrum = Spectrum.from_lists(bin_centers, fy_list=bin_values_fy)
    assert spectrum.num_bins == len(bin_centers)
    assert np.array_equal(spectrum.bin_values_fy, spectrum_fig3p3_olko_phd.bin_values_fy)
    assert np.array_equal(spectrum.yF, spectrum_fig3p3_olko_phd.yF)
    assert np.array_equal(spectrum.fy, spectrum_fig3p3_olko_phd.fy)
    assert np.array_equal(spectrum.yfy, spectrum_fig3p3_olko_phd.yfy)
    assert np.array_equal(spectrum.ydy, spectrum_fig3p3_olko_phd.ydy)

def test_invalid_initialization():
    y_list = [1, 2, 3, 4, 5]
    fy_list = [3, 6, 2, 8, 5]
    yfy_list = [2, 4, 6, 8, 10]

    # Trying to initialize with conflicting values
    with pytest.raises(ValueError):
        Spectrum.from_lists(y_list=y_list, fy_list=fy_list, yfy_list=yfy_list)

    # Trying to initialize with missing values
    with pytest.raises(ValueError):
        Spectrum.from_lists(y_list)


def test_if_printout_has_multiple_lines(small_spectrum: Spectrum, capsys):
    print(small_spectrum)
    captured = capsys.readouterr()
    output_lines = captured.out.splitlines()
    assert len(output_lines) > 1

def test_loading_from_str_with_fy():
    empty_str = ""
    spectrum = from_str(empty_str)
    assert spectrum.num_bins == 0
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

def test_bin_centers_not_sorted():
    y_list = [1, 2, 3, 4, 3.5]
    fy_list = [3, 6, 2, 8, 5]
    with pytest.raises(ValueError):
        Spectrum.from_lists(y_list, fy_list=fy_list)