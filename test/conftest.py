import numpy as np
from numpy.typing import NDArray
import pytest
from src.spectrum import SpectrumData

@pytest.fixture
def spectrum_fig3p3_olko_phd() -> SpectrumData:
    bin_centers = [1,2,3]
    bin_values = [2,2,2]
    return SpectrumData.from_lists(y_list=bin_centers, fy_list=bin_values)

@pytest.fixture
def small_spectrum() -> SpectrumData:
    bin_centers = [1, 2, 3, 4]
    bin_values_fy = [0.1, 0.2, 0.3, 0.4]
    return SpectrumData.from_lists(y_list=bin_centers, fy_list=bin_values_fy)

@pytest.fixture
def not_normalised_spectrum() -> SpectrumData:
    bin_centers = [1, 2, 3, 4]
    bin_values_fy = [1, 2, 3, 4]
    return SpectrumData.from_lists(y_list=bin_centers, fy_list=bin_values_fy)

@pytest.fixture
def spectrum_log_binning() -> SpectrumData:
    bin_centers = [0.1, 1, 10, 100]
    bin_values_fy = [0.1, 0.2, 0.3, 0.4]
    return SpectrumData.from_lists(y_list=bin_centers, fy_list=bin_values_fy)


@pytest.fixture
def spectrum_unknown_binning() -> SpectrumData:
    bin_centers = np.array([1,2,4,5])
    bin_values_fy = np.array([0.2, 0.2, 0.2, 0.4])
    return SpectrumData(bin_centers=bin_centers, bin_values_fy=bin_values_fy)

@pytest.fixture
def step_function_with_factor(request) -> float:
    user_factor = request.node.get_closest_marker("factor")
    factor = 1
    if user_factor:
        factor = user_factor.args[0]
    def _step_function(x: float) -> float:
        if x < 0:
            return 0
        if x > 2:
            return 0
        return 0.5*factor
    return _step_function

@pytest.fixture
def step_functions_with_factor(request) -> float:
    user_factor = request.node.get_closest_marker("factor")
    factor = 1
    if user_factor:
        factor = user_factor.args[0]
    def _step_functions(x: NDArray) -> NDArray:
        '''Array version of step_function'''
        result = 0.5 * factor * np.ones_like(x)
        result[x < 0] = 0
        result[x > 2] = 0
        return result
    return _step_functions