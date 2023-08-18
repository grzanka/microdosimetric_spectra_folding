import logging
from numpy.typing import NDArray

def check_if_only_one_initialized(fy: NDArray, yfy: NDArray, ydy: NDArray):
    '''Check if the two arrays are not None and have the same size.'''    
    if fy.size == 0 and yfy.size != 0 and ydy.size != 0 \
        or fy.size != 0 and yfy.size == 0 and ydy.size != 0 \
        or fy.size != 0 and yfy.size != 0 and ydy.size == 0:
            raise ValueError("Only one of fy, yfy, ydy must be initialized (not two)")
    if fy.size != 0 and yfy.size != 0 and ydy.size != 0:
        raise ValueError("Only one of fy, yfy, ydy must be initialized (not three)")
    
def check_if_array_holds_spectrum(data_array: NDArray):
    if data_array.size == 0:
        raise ValueError("data_string must contain at least one row")
    if data_array.ndim != 2:
        logging.debug("data_array.ndim is {}".format(data_array.ndim))
        raise ValueError("data_string must contain two columns")
    if data_array.shape[1] != 2:
        logging.debug("data_array.shape is {}".format(data_array.shape))
        raise ValueError("data_string must contain two columns")

def check_if_same_length_as_bin_centers(bin_centers: NDArray, fy: NDArray, yfy: NDArray, ydy: NDArray):
    '''Check if the two arrays are not None and have the same size.'''
    if fy.size != bin_centers.size:
        raise ValueError("fy must have the same size as bin_centers")
    if yfy.size != bin_centers.size:
        raise ValueError("yfy must have the same size as bin_centers")
    if ydy.size != bin_centers.size:
        raise ValueError("ydy must have the same size as bin_centers")