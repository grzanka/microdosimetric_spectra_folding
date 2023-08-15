import numpy as np
import pytest
from src.spectrum import Spectrum  # Assuming your class is defined in a file named 'spectrum.py'

def test_creation_from_lists():
    bin_centers = [1, 2, 3, 4, 5]
    bin_values_f = [3, 6, 2, 8, 5]

    spectrum = Spectrum.from_lists(bin_centers, bin_values_list=bin_values_f)
    assert spectrum.num_bins == len(bin_centers)
    assert np.array_equal(spectrum.bin_values_f, np.array(bin_values_f))
    assert np.array_equal(spectrum.bin_values_yfy, np.array(bin_values_f) * np.array(bin_centers))
    assert np.array_equal(spectrum.bin_values_ydy, np.array(bin_values_f) * np.array(bin_centers) ** 2)

def test_invalid_initialization():
    with pytest.raises(ValueError):
        Spectrum()

    bin_centers = [1, 2, 3, 4, 5]
    bin_values_f = [3, 6, 2, 8, 5]
    bin_values_yfy = [2, 4, 6, 8, 10]
    bin_values_ydy = [0.5, 2, 4.5, 8, 12.5]

    # Trying to initialize with conflicting values
    with pytest.raises(ValueError):
        Spectrum.from_lists(bin_centers, bin_values_list=bin_values_f, bin_values_yfy_list=bin_values_yfy)

    # Trying to initialize with missing values
    with pytest.raises(ValueError):
        Spectrum.from_lists(bin_centers)

def test_arithmetic_operations():
    bin_centers = [1, 2, 3, 4, 5]
    bin_values_f = [3, 6, 2, 8, 5]

    spectrum = Spectrum.from_lists(bin_centers, bin_values_list=bin_values_f)

    assert np.array_equal(spectrum.bin_values_yfy, np.array(bin_values_f) * np.array(bin_centers))
    assert np.array_equal(spectrum.bin_values_ydy, np.array(bin_values_f) * np.array(bin_centers) ** 2)

    spectrum = Spectrum.from_lists(bin_centers, bin_values_yfy_list=spectrum.bin_values_yfy)
    assert np.array_equal(spectrum.bin_values_f, np.array(bin_values_f))
    assert np.array_equal(spectrum.bin_values_ydy, np.array(bin_values_f) * np.array(bin_centers) ** 2)

    spectrum = Spectrum.from_lists(bin_centers, bin_values_ydy_list=spectrum.bin_values_ydy)
    assert np.array_equal(spectrum.bin_values_f, np.array(bin_values_f))
    assert np.array_equal(spectrum.bin_values_yfy, np.array(bin_values_f) * np.array(bin_centers))


if __name__ == "__main__":
    pytest.main()
