from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

def step_function(x: float, norm: float = 1.) -> float:
    if x < 0:
        return 0
    if x > 2:
        return 0
    return 0.5*norm

def step_functions(x: NDArray, norm: float = 1.) -> NDArray:
    '''Array version of step_function'''
    result = 0.5 * norm * np.ones_like(x)
    result[x < 0] = 0
    result[x > 2] = 0
    return result

def function_norm(func: Callable, lower_limit : float, upper_limit : float, args: tuple = ()) -> float:
  I = quad(func=func, a=lower_limit, b=upper_limit, args=args, limit=500, epsabs=1e-5, epsrel=1e-4)
  return I[0]