from enum import Enum, auto
from typing import Callable, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad


def step_function(
    x: float, norm: float = 1.0, lower: float = 0, upper: float = 2, zero: float = 0.0
) -> float:
    if x < lower:
        return zero
    if x > upper:
        return zero
    return norm / (upper - lower)


def step_functions(
    x: NDArray, norm: float = 1.0, lower: float = 0, upper: float = 2, zero: float = 0.0
) -> NDArray:
    """Array version of step_function"""
    result = norm / (upper - lower) * np.ones_like(x)
    result[x < lower] = zero
    result[x > upper] = zero
    return result


def function_norm(
    func: Callable,
    lower_limit: float = -np.inf,
    upper_limit: float = np.inf,
    args: tuple = (),
    kwargs: dict = {},
    include_error: bool = False,
) -> Union[float, Tuple[float, float]]:
    I = quad(func=func, a=lower_limit, b=upper_limit, args=args, **kwargs)
    result = I[0]
    if include_error:
        result = I
    return result


def self_convolution_integrand(func: Callable) -> Callable:
    """Return the integrand of the convolution integral of func with itself."""

    def _convolution_integrand(
        t: float, y: float, args: tuple = (), kwargs: dict = {}
    ) -> float:
        return func(t, *args, **kwargs) * func(y - t, *args, **kwargs)

    return _convolution_integrand


def self_convolution(
    func: Callable,
    lower_limit: float = -np.inf,
    upper_limit: float = np.inf,
    args: tuple = (),
    kwargs: dict = {},
) -> Callable:
    """Return the convolution of func with itself."""

    def _convolution(y: float, integrand_args: tuple = ()) -> float:
        integrand = self_convolution_integrand(func)
        return quad(
            func=integrand,
            a=lower_limit,
            b=upper_limit,
            args=(y, integrand_args),
            *args,
            **kwargs
        )

    return _convolution


def convolution_integrand(first_func: Callable, second_func: Callable) -> Callable:
    """Return the integrand of the convolution integral of func with itself."""

    def _convolution_integrand(
        t: float, y: float, args: tuple = (), kwargs: dict = {}
    ) -> float:
        return first_func(t, *args, **kwargs) * second_func(y - t, *args, **kwargs)

    return _convolution_integrand


def convolution(
    first_func: Callable,
    second_func: Callable,
    lower_limit: float = -np.inf,
    upper_limit: float = np.inf,
    args: tuple = (),
    kwargs: dict = {},
) -> Callable:
    """Return the convolution of two functions."""

    def _convolution(y: float, integrand_args: tuple = ()) -> float:
        integrand = convolution_integrand(first_func, second_func)
        return quad(
            func=integrand,
            a=lower_limit,
            b=upper_limit,
            args=(y, integrand_args),
            *args,
            **kwargs
        )

    return _convolution
