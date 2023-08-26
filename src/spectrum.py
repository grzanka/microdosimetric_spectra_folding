from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from src.helpers import SpectrumBinningType, bin_edges, binning_type, first_moment, SpectrumValueType, others_from_y_and_fy, others_from_y_and_yfy
from src.checks import check_if_array_holds_spectrum, check_if_bin_centers_valid, check_if_only_one_initialized, check_if_same_length_as_bin_centers

@dataclass(frozen=True)
class SpectrumData:
    '''Spectrum class. It is immutable. It can be initialized from bin_centers and one of bin_values_fy, bin_values_yfy, bin_values_ydy.
    
    z [ Gy ] = 0.204 * y [keV/um] / d^2 [um]
    '''

    bin_centers: NDArray = field(default_factory=lambda: np.empty(0))

    bin_values_fy: NDArray = field(default_factory=lambda: np.empty(0))
    bin_values_yfy: NDArray = field(default_factory=lambda: np.empty(0))
    bin_values_ydy: NDArray = field(default_factory=lambda: np.empty(0))

    bin_values_dy: NDArray = field(default_factory=lambda: np.empty(0))

    bin_values_fy_normalized: NDArray = field(default_factory=lambda: np.empty(0))
    bin_values_yfy_normalized: NDArray = field(default_factory=lambda: np.empty(0))
    bin_values_dy_normalized: NDArray = field(default_factory=lambda: np.empty(0))
    bin_values_ydy_normalized: NDArray = field(default_factory=lambda: np.empty(0))

    binning_type: SpectrumBinningType = SpectrumBinningType.unknown

    bin_edges: NDArray = field(default_factory=lambda: np.empty(0))
    bin_widths: NDArray = field(default_factory=lambda: np.empty(0))
    
    bin_nums: int = 0
    norm: float = np.nan
    yF: float = np.nan
    yD: float = np.nan

    def __post_init__(self):
        check_if_bin_centers_valid(self.bin_centers)
        object.__setattr__(self, 'num_bins', self.bin_centers.size)

        check_if_only_one_initialized(self.bin_values_fy, self.bin_values_yfy, self.bin_values_ydy)

        fy = dy = yfy = ydy = np.empty(0)
        if self.bin_values_fy.size != 0:
            fy = self.bin_values_fy
            yfy, dy, ydy = others_from_y_and_fy(y=self.bin_centers, fy=self.bin_values_fy)
        if self.bin_values_yfy.size != 0:
            yfy = self.bin_values_yfy
            fy, dy, ydy = others_from_y_and_yfy(y=self.bin_centers, yfy=yfy)
        if self.bin_values_ydy.size != 0 or self.bin_values_dy.size != 0:
            raise NotImplementedError("deriving spectrum from dy or ydy is not implemented yet")
        if self.bin_values_fy.size == 0:
            object.__setattr__(self, 'bin_values_fy', fy)
        if self.bin_values_yfy.size == 0:
            object.__setattr__(self, 'bin_values_yfy', yfy)
        object.__setattr__(self, 'bin_values_dy', dy)
        object.__setattr__(self, 'bin_values_ydy', ydy)

        check_if_same_length_as_bin_centers(bin_centers=self.bin_centers, fy=self.bin_values_fy, yfy=self.bin_values_yfy, ydy=self.bin_values_ydy)

        # Set binning type
        object.__setattr__(self, 'binning_type', binning_type(self.bin_centers))

        # Set bin edges and bin widths
        object.__setattr__(self, 'bin_edges', bin_edges(self.bin_centers, self.binning_type))
        object.__setattr__(self, 'bin_widths', np.diff(self.bin_edges))

        # set means
        if self.bin_values_fy.size > 0:
            object.__setattr__(self, 'yF', first_moment(bin_edges=self.bin_edges, bin_values=self.bin_values_fy))
        if self.bin_values_dy.size > 0:
            object.__setattr__(self, 'yD', first_moment(bin_edges=self.bin_edges, bin_values=self.bin_values_dy))

        # set normalized values if bin_centers are initialized
        object.__setattr__(self, 'norm', self.fy @ self.bin_widths)
        fy_norm = self.fy / self.norm if self.norm != 0 else np.zeros_like(self.fy)
        object.__setattr__(self, 'bin_values_fy_normalized', fy_norm)
        yfy_norm, dy_norm, ydy_norm = others_from_y_and_fy(y=self.bin_centers, fy=self.bin_values_fy_normalized)
        object.__setattr__(self, 'bin_values_yfy_normalized', yfy_norm)
        object.__setattr__(self, 'bin_values_dy_normalized', dy_norm)
        object.__setattr__(self, 'bin_values_ydy_normalized', ydy_norm)

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
    def from_lists(cls, y_list : list=[], fy_list : list =[], yfy_list: list=[], ydy_list: list=[]):
        y_array = np.array(y_list)
        fy_array = np.array(fy_list) if fy_list else np.empty(0)
        yfy_array = np.array(yfy_list) if yfy_list else np.empty(0)
        ydy_array = np.array(ydy_list) if ydy_list else np.empty(0)
        return cls(bin_centers = y_array, bin_values_fy=fy_array, bin_values_yfy=yfy_array, bin_values_ydy=ydy_array)

    def __str__(self):
        fields = [(name, value) for name, value in self.__dict__.items() if isinstance(value, np.ndarray)]

        output = ""
        for field_name, field_value in fields:
            output += f"{field_name}:\n{field_value}\n\n"
        
        return output


@dataclass(frozen=True)
class LinealEnergySpectrum(SpectrumData):
    data: SpectrumData = field(default_factory=lambda: SpectrumData())


@dataclass(frozen=True)
class SpecificEnergySpectrum(SpectrumData):
    data: SpectrumData = field(default_factory=lambda: SpectrumData())


def from_array(data_array: NDArray, value_type: SpectrumValueType = SpectrumValueType.yfy) -> SpectrumData:
    '''Load spectrum from array. The array must contain two columns: bin_centers and bin_values_fy.'''
    check_if_array_holds_spectrum(data_array)
    result = SpectrumData()
    if value_type == SpectrumValueType.fy:
        result = SpectrumData(bin_centers=data_array[:,0], bin_values_fy=data_array[:,1])
    elif value_type == SpectrumValueType.yfy:
        result = SpectrumData(bin_centers=data_array[:,0], bin_values_yfy=data_array[:,1])
    elif value_type == SpectrumValueType.ydy:
        result = SpectrumData(bin_centers=data_array[:,0], bin_values_ydy=data_array[:,1])
    return result

def from_str(data_string : str, value_type: SpectrumValueType = SpectrumValueType.yfy, **kwargs) -> SpectrumData:
    '''Load spectrum from string. The string must contain two columns: bin_centers and bin_values_fy.'''
    if data_string:
        data_array = np.genfromtxt(StringIO(data_string), **kwargs)
        check_if_array_holds_spectrum(data_array)
        result = from_array(data_array, value_type)
    else:
        result = SpectrumData()
    return result

def from_csv(file_path: Path, value_type: SpectrumValueType = SpectrumValueType.yfy, **kwargs) -> SpectrumData:
    '''Load spectrum from csv file. The file must contain two columns: bin_centers and bin_values_fy.'''
    data_array = np.genfromtxt(file_path, **kwargs)
    check_if_array_holds_spectrum(data_array)
    result = from_array(data_array, value_type)
    return result
