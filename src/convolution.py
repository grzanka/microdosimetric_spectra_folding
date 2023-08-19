from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

def step_function(x: float, norm: float = 1., lower: float = 0, upper: float = 2) -> float:
    if x < lower:
        return 0
    if x > upper:
        return 0
    return norm / (upper - lower)

def step_functions(x: NDArray, norm: float = 1., lower: float = 0, upper: float = 2) -> NDArray:
    '''Array version of step_function'''
    result = norm / (upper - lower) * np.ones_like(x)
    result[x < lower] = 0
    result[x > upper] = 0
    return result

def function_norm(func: Callable, lower_limit : float = -np.inf, upper_limit : float = np.inf, args: tuple = ()) -> float:
  I = quad(func=func, a=lower_limit, b=upper_limit, args=args)
  return I[0]

def convolution_integrand(func: Callable) -> Callable:
  '''Return the integrand of the convolution integral of func with itself.'''
  def _convolution_integrand(t: float, y: float, args: tuple = (), kwargs: dict = {}) -> float:
    return func(t, *args, **kwargs) * func(y-t, *args, **kwargs)
  return _convolution_integrand

def convolution(func: Callable, lower_limit : float = -np.inf, upper_limit : float = np.inf) -> Callable:
  '''Return the convolution of func with itself.'''  
  def _convolution(y: float, args: tuple = (), kwargs: dict = {}) -> float:
    integrand = convolution_integrand(func)
    return quad(func=integrand, a=lower_limit, b=upper_limit, args=(y, *args), limit=500, epsabs=1e-5, epsrel=1e-4)[0]
  return _convolution