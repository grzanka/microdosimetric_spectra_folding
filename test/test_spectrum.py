import numpy as np
import pytest
from src.spectrum import SpectrumData, SpectrumBinningType, from_str


def test_empty_spectrum():
    empty_spectrum = SpectrumData()
    assert empty_spectrum.num_bins == 0
    assert empty_spectrum.binning_type == SpectrumBinningType.unknown


def test_negative_spectrum():
    with pytest.raises(ValueError):
        SpectrumData.from_lists([-1, 0, 1], [-1, -2, -3])


def test_spectrum_with_one_bin():
    spectrum = SpectrumData.from_lists([1], [1])
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
    spectrum = SpectrumData.from_lists([1, 2, 3], [0, 0, 0])
    assert spectrum.num_bins == 3
    assert spectrum.binning_type == SpectrumBinningType.linear
    assert np.isnan(
        spectrum.freq_mean
    ), "mean freq must be NaN if bin_values are all zero"


def test_creation_from_lists(spectrum_data_fig3p3_olko_phd: SpectrumData):
    bin_centers = spectrum_data_fig3p3_olko_phd.bin_centers.tolist()
    bin_values_freq = spectrum_data_fig3p3_olko_phd.bin_values_freq.tolist()

    spectrum = SpectrumData.from_lists(bin_centers, freq=bin_values_freq)
    assert spectrum.num_bins == len(bin_centers)
    assert np.array_equal(
        spectrum.bin_values_freq, spectrum_data_fig3p3_olko_phd.bin_values_freq
    )
    assert np.array_equal(spectrum.freq_mean, spectrum_data_fig3p3_olko_phd.freq_mean)
    assert np.array_equal(
        spectrum.bin_values_freq_times_x,
        spectrum_data_fig3p3_olko_phd.bin_values_freq_times_x,
    )
    assert np.array_equal(
        spectrum.bin_values_dose_times_x,
        spectrum_data_fig3p3_olko_phd.bin_values_dose_times_x,
    )


def test_invalid_initialization():
    x_list = [1, 2, 3, 4, 5]
    freq_list = [3, 6, 2, 8, 5]
    freq_times_x_list = [2, 4, 6, 8, 10]

    # Trying to initialize with conflicting values
    with pytest.raises(ValueError):
        SpectrumData.from_lists(
            x=x_list, freq=freq_list, freq_times_x=freq_times_x_list
        )

    # Trying to initialize with missing values
    with pytest.raises(ValueError):
        SpectrumData.from_lists(x_list)


def test_if_printout_has_multiple_lines(small_spectrum: SpectrumData, capsys):
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
    x_list = [1, 2, 3, 4, 3.5]
    freq_list = [3, 6, 2, 8, 5]
    with pytest.raises(ValueError):
        SpectrumData.from_lists(x_list, freq=freq_list)
