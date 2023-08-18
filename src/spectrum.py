from dataclasses import dataclass, field
from enum import Enum, auto
from io import StringIO
import logging
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from src.helpers import SpectrumBinningType, bin_edges, binning_type, first_moment, SpectrumValueType
from src.checks import check_if_array_holds_spectrum, check_if_only_one_initialized

@dataclass(frozen=True)
class Spectrum:
    '''Spectrum class. It is immutable. It can be initialized from bin_centers and one of bin_values_fy, bin_values_yfy, bin_values_ydy.'''

    bin_centers: NDArray = np.empty(0)

    bin_values_fy: NDArray = np.empty(0)
    bin_values_yfy: NDArray = np.empty(0)
    bin_values_ydy: NDArray = np.empty(0)

    bin_values_dy: NDArray = np.empty(0)

    bin_values_fy_normalized: NDArray = np.empty(0)
    bin_values_yfy_normalized: NDArray = np.empty(0)
    bin_values_dy_normalized: NDArray = np.empty(0)
    bin_values_ydy_normalized: NDArray = np.empty(0)

    binning_type: SpectrumBinningType = SpectrumBinningType.unknown

    bin_edges: NDArray = np.empty(0)
    bin_widths: NDArray = np.empty(0)
    bin_nums: int = 0

    def __post_init__(self):
        check_if_only_one_initialized(self.bin_values_fy, self.bin_values_yfy, self.bin_values_ydy)


        fy_initialized = self.bin_values_fy.size != 0
        yfy_initialized = self.bin_values_yfy.size != 0
        ydy_initialized = self.bin_values_ydy.size != 0
        if fy_initialized:
            logging.debug("bin_values_fy is initialized to {}".format(self.bin_values_fy))
            # yfy = y * f(y)
            object.__setattr__(self, 'bin_values_yfy', self.y * self.fy)
            logging.debug("bin_values_yfy is initialized to {}".format(self.bin_values_yfy))
            
            # d(y) = (y / yF) * f(y)
            object.__setattr__(self, 'bin_values_dy', (self.y / self.yF) * self.fy)
            logging.debug("bin_values_dy is initialized to {}".format(self.bin_values_dy))
            object.__setattr__(self, 'bin_values_ydy', self.y * self.dy)
            logging.debug("bin_values_ydy is initialized to {}".format(self.bin_values_ydy))

        if yfy_initialized:
            logging.debug("bin_values_yfy is initialized to {}".format(self.bin_values_yfy))
            # yfy = y * f(y)   =>    fy = yfy / y
            object.__setattr__(self, 'bin_values_fy', self.yfy / self.y)

            # d(y) = (y / yF) * f(y)
            object.__setattr__(self, 'bin_values_dy', (self.y / self.yF) * self.fy)
            object.__setattr__(self, 'bin_values_ydy', self.y * self.dy)
        if ydy_initialized:
            raise NotImplementedError("deriving spectrum from ydy is not implemented yet")

        # if bin values are initialized then sum of bin_values_fy must be positive
        if self.bin_values_fy.size > 0 and self.bin_values_fy.sum() <= 0:
            raise ValueError("Sum of bin_values_f must be positive")

        if len(self.bin_centers) != len(self.bin_values_fy) \
                or len(self.bin_centers) != len(self.bin_values_yfy) \
                or len(self.bin_centers) != len(self.bin_values_ydy):
            raise ValueError("All arrays must have the same size")

        # check if bin_centers are sorted
        if not np.all(np.diff(self.bin_centers) > 0):
            raise ValueError("bin_centers must be sorted")

        object.__setattr__(self, 'num_bins', self.bin_centers.size)

        # Set binning type
        object.__setattr__(self, 'binning_type', binning_type(self.bin_centers))

        # Set bin edges and bin widths
        object.__setattr__(self, 'bin_edges', bin_edges(self.bin_centers, self.binning_type))
        object.__setattr__(self, 'bin_widths', np.diff(self.bin_edges))

        # set normalized values if bin_centers are initialized
        if self.bin_centers.size > 0 and self.bin_values_fy.size > 0:
            logging.debug("self.fy is initialized to {}".format(self.fy))
            logging.debug("self.fy.sum() is initialized to {}".format(self.fy.sum()))
            logging.debug("self.norm is initialized to {}".format(self.norm))
            object.__setattr__(self, 'bin_values_fy_normalized', self.fy / self.norm)
            logging.debug("bin_values_fy_normalized is initialized to {}".format(self.bin_values_fy_normalized))
            logging.debug("(self.y / self.yF) is initialized to {}".format((self.y / self.yF)))
            logging.debug("self.fy_norm is initialized to {}".format((self.fy_norm)))
            object.__setattr__(self, 'bin_values_dy_normalized', (self.y / self.yF) * self.bin_values_fy_normalized)
            object.__setattr__(self, 'bin_values_yfy_normalized', self.y * self.fy_norm)
            object.__setattr__(self, 'bin_values_ydy_normalized', self.y * self.bin_values_dy_normalized)

    @property
    def f_sum(self) -> float:
        '''Sum of bin_values_fy. It is equal to 1 if the spectrum is normalized and has bin widths = 1.'''
        return self.fy.sum()
    
    @property
    def norm(self) -> float:
        '''Normalization factor. Defined as integral of fy over all bins. It is equal to 1 if the spectrum is normalized (for lin or log binning).'''
        logging.debug("self.bin_widths is {}".format(self.bin_widths))
        logging.debug("self.fy is {}".format(self.fy))        
        result = self.fy @ self.bin_widths
        logging.debug("result is {}".format(result))
        return result
    
    @property
    def yF(self) -> float:
        return first_moment(bin_centers=self.y, bin_values=self.fy)
    
    @property
    def yD(self) -> float:
        return 0
    
    @property
    def y(self) -> NDArray:
        return self.bin_centers
    
    @property
    def fy(self) -> NDArray:
        return self.bin_values_fy
    
    @property
    def dy(self) -> NDArray:
        return self.bin_values_dy

    @property
    def yfy(self) -> NDArray:
        return self.bin_values_yfy
    
    @property
    def ydy(self) -> NDArray:
        return self.bin_values_ydy

    @property
    def fy_norm(self) -> NDArray:
        return self.bin_values_fy_normalized

    @property
    def dy_norm(self) -> NDArray:
        return self.bin_values_dy_normalized

    @property
    def yfy_norm(self) -> NDArray:
        return self.bin_values_yfy_normalized
    
    @property
    def ydy_norm(self) -> NDArray:
        return self.bin_values_ydy_normalized
    
    def bin_numbers(self, y : NDArray) -> NDArray:
        '''Return the indices of the bins to which each value in input array belongs.'''
        return np.digitize(x=y, bins=self.bin_edges) - 1
    
    def bin_number(self, y : float) -> float:
        '''Return the index of the bins to which the value belongs.'''
        return self.bin_numbers(np.array([y]))[0]
    
    def bin_values(self, y : NDArray, spectrum_value_type: SpectrumValueType = SpectrumValueType.yfy) -> float:
        '''Return the value of the bin to which the value belongs.'''   
        ind = self.bin_numbers(y)
        ind[ind < 0] = -1
        ind[ind >= self.bin_centers.size] = -1
        bin_values_extended = np.zeros(shape=(self.bin_centers.size+2,))
        if spectrum_value_type == SpectrumValueType.fy:
            bin_values_extended[1:-1] = self.fy
        elif spectrum_value_type == SpectrumValueType.yfy:
            bin_values_extended[1:-1] = self.yfy
        elif spectrum_value_type == SpectrumValueType.ydy:
            bin_values_extended[1:-1] = self.ydy
        return bin_values_extended.take(indices=ind+1, mode='clip')
    
    def bin_value(self, y : float, spectrum_value_type: SpectrumValueType = SpectrumValueType.yfy) -> float:
        '''Return the value of the bin to which the value belongs.'''
        return self.bin_values(np.array([y]), spectrum_value_type)[0]
    
    @classmethod
    def from_lists(cls, bin_centers_list : list=[], bin_values_list : list =[], bin_values_yfy_list: list=[], bin_values_ydy_list: list=[]):
        bin_centers = np.array(bin_centers_list)
        bin_values_fy = np.array(bin_values_list) if bin_values_list else np.empty(0)
        bin_values_yfy = np.array(bin_values_yfy_list) if bin_values_yfy_list else np.empty(0)
        bin_values_ydy = np.array(bin_values_ydy_list) if bin_values_ydy_list else np.empty(0)
        return cls(bin_centers = bin_centers, bin_values_fy=bin_values_fy, bin_values_yfy=bin_values_yfy, bin_values_ydy=bin_values_ydy)

    def __str__(self):
        fields = [(name, value) for name, value in self.__dict__.items() if isinstance(value, np.ndarray)]

        output = ""
        for field_name, field_value in fields:
            output += f"{field_name}:\n{field_value}\n\n"
        
        return output
    
def from_array(data_array: NDArray, value_type: SpectrumValueType = SpectrumValueType.yfy) -> Spectrum:
    '''Load spectrum from array. The array must contain two columns: bin_centers and bin_values_fy.'''
    check_if_array_holds_spectrum(data_array)
    result = Spectrum()
    if value_type == SpectrumValueType.fy:
        result = Spectrum(bin_centers=data_array[:,0], bin_values_fy=data_array[:,1])
    elif value_type == SpectrumValueType.yfy:
        result = Spectrum(bin_centers=data_array[:,0], bin_values_yfy=data_array[:,1])
    elif value_type == SpectrumValueType.ydy:
        result = Spectrum(bin_centers=data_array[:,0], bin_values_ydy=data_array[:,1])
    return result

def from_str(data_string : str, value_type: SpectrumValueType = SpectrumValueType.yfy, **kwargs) -> Spectrum:
    '''Load spectrum from string. The string must contain two columns: bin_centers and bin_values_fy.'''
    data_array = np.genfromtxt(StringIO(data_string), **kwargs)
    check_if_array_holds_spectrum(data_array)
    result = from_array(data_array, value_type)
    return result

def from_csv(file_path: Path, value_type: SpectrumValueType = SpectrumValueType.yfy, **kwargs) -> Spectrum:
    '''Load spectrum from csv file. The file must contain two columns: bin_centers and bin_values_fy.'''
    data_array = np.genfromtxt(file_path, **kwargs)
    check_if_array_holds_spectrum(data_array)
    result = from_array(data_array, value_type)
    return result
