import pytest
import numpy as np
from src.convolution.numerical import step_function, step_functions, function_norm

def test_step_function(step_functions_with_factor, step_function_with_factor):
    assert step_function_with_factor(-1) == 0
    xvalues = np.linspace(start=-1, stop=3, num=30)
    expected_yvalues = step_functions_with_factor(xvalues)
    assert np.array_equal(step_functions(xvalues), expected_yvalues)

@pytest.mark.factor(5)
def test_step_function_with_factor(step_functions_with_factor, step_function_with_factor):
    assert step_function_with_factor(-1) == 0
    xvalues = np.linspace(start=-1, stop=3, num=30)
    expected_yvalues = step_functions_with_factor(xvalues)
    assert np.array_equal(step_functions(xvalues, norm=5), expected_yvalues)

def test_norm_of_step_function():
    assert function_norm(step_function, lower_limit=-1, upper_limit=3) == pytest.approx(1)
    assert function_norm(step_function) == pytest.approx(1)
    _step_function_derived = lambda x: step_functions(np.array([x]))[0]
    assert function_norm(_step_function_derived, lower_limit=-1, upper_limit=3) == pytest.approx(1)
    assert function_norm(_step_function_derived) == pytest.approx(1)
    # check norm with factor
    assert function_norm(step_function, lower_limit=-1, upper_limit=3, args=(5,)) == pytest.approx(5)
    _step_function_derived_with_param = lambda x,y: step_functions(np.array([x]), norm=y)[0]
    assert function_norm(_step_function_derived_with_param, lower_limit=-1, upper_limit=3, args=(7,)) == pytest.approx(7)

    assert function_norm(step_function, args=(1, -0.5, 0.5)) == pytest.approx(1)
    assert function_norm(step_function, args=(1, -500, 500)) == pytest.approx(1)
