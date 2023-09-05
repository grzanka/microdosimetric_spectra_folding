import numpy as np
from tqdm import tqdm
from src.convolution.numerical import function_norm
from src.helpers import SpectrumValueType
from src.spectrum import SpectrumData


def cfd(
    x: float, data: SpectrumData, include_error: bool = False, kwargs: dict = {}
) -> float:
    pdf_function = lambda x: data.bin_value(x, spectrum_value_type=SpectrumValueType.fy)
    result = function_norm(
        pdf_function,
        lower_limit=data.bin_edges[0],
        upper_limit=x,
        include_error=include_error,
        kwargs=kwargs,
    )
    return result


def cfds(
    x: np.ndarray, data: SpectrumData, include_error: bool = False, kwargs: dict = {}
) -> np.ndarray:
    result = np.array(
        [cfd(xi, data, include_error=include_error, kwargs=kwargs) for xi in x]
    )
    return result


def cfds_with_progress(
    x: np.ndarray, data: SpectrumData, include_error: bool = False, kwargs: dict = {}
) -> np.ndarray:
    result = []
    for xi in tqdm(x, desc="Processing"):
        result.append(cfd(xi, data, include_error=include_error, kwargs=kwargs))
    return np.array(result)
