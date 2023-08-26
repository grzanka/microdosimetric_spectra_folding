import numpy as np
import pytest
from src.spectrum import SpectrumData, SpectrumValueType

def test_y_values_fun(small_spectrum: SpectrumData):
    # bin centers 1 2 3 4
    # bin edges 0.5 1.5 2.5 3.5 4.5
    # fy: 0.1, 0.2, 0.3, 0.4
    # bins:
    # as -1: <-inf, 0.5)
    # 0: [0.5, 1.5)
    # 1: [1.5, 2.5)
    # 2: [2.5, 3.5)
    # 3: [3.5, 4.5)
    # as 4: [4.5, inf)
    y_values = np.array([
        -1, 0, 0.3,  # bin -1, value 0
        0.5, 1, 1.2, # bin 0, value 0.1
        1.5, 2.3,  # bin 1, value 0.2
        2.5, 3, # bin 2, value 0.3
        3.5, 4, # bin 3, value 0.4
        4.5, 5 # bin 4, value 0
        ]) 
    expected_fy_values = np.array([
        0, 0, 0, 
        0.1, 0.1, 0.1,
        0.2, 0.2,
        0.3, 0.3,
        0.4, 0.4,
        0, 0
        ])
    rebinned_fy_values = small_spectrum.bin_values(y=y_values, spectrum_value_type=SpectrumValueType.fy)
    assert np.array_equal(rebinned_fy_values, expected_fy_values)
    assert small_spectrum.bin_value(y=0.3, spectrum_value_type=SpectrumValueType.fy) == pytest.approx(0)
    assert small_spectrum.bin_value(y=0.5, spectrum_value_type=SpectrumValueType.fy) == pytest.approx(0.1)