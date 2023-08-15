from dataclasses import dataclass, field
import logging
import numpy as np
from numpy.typing import NDArray


def first_moment(bin_centers: NDArray, bin_values: NDArray) -> float:
    '''Calculate the first moment of a spectrum. It may be not normalized.'''
    if bin_values.sum() == 0:
        raise ZeroDivisionError("Sum of bin_values must be positive")
    return np.sum(bin_centers * bin_values) / np.sum(bin_values)

@dataclass(frozen=True)
class Spectrum:

    # bin centers
    bin_centers: np.array = field(default_factory=lambda: np.empty(0))

    # default bin values
    bin_values_fy: np.array = field(default_factory=lambda: np.empty(0))
    bin_values_yfy: np.array = field(default_factory=lambda: np.empty(0))
    bin_values_ydy: np.array = field(default_factory=lambda: np.empty(0))

    # derived values
    bin_values_dy: np.array = field(default_factory=lambda: np.empty(0))

    # normalized values
    bin_values_fy_normalized: np.array = field(default_factory=lambda: np.empty(0))
    bin_values_yfy_normalized: np.array = field(default_factory=lambda: np.empty(0))
    bin_values_dy_normalized: np.array = field(default_factory=lambda: np.empty(0))
    bin_values_ydy_normalized: np.array = field(default_factory=lambda: np.empty(0))

    def __post_init__(self):
        if self.bin_centers.size == 0:
            raise ValueError("bin_centers must be initialized")
        
        if self.bin_values_fy.size == 0 and self.bin_values_yfy.size == 0 and self.bin_values_ydy.size == 0:
            raise ValueError("At least one of bin_values_fy, bin_values_yfy, bin_values_ydy must be initialized (not zero)")
        if self.bin_values_fy.size == 0 and self.bin_values_yfy.size != 0 and self.bin_values_ydy.size != 0 \
                or self.bin_values_fy.size != 0 and self.bin_values_yfy.size == 0 and self.bin_values_ydy.size != 0 \
                or self.bin_values_fy.size != 0 and self.bin_values_yfy.size != 0 and self.bin_values_ydy.size == 0:
            raise ValueError("Only one of bin_values_fy, bin_values_yfy, bin_values_ydy must be initialized (not two)")
        if self.bin_values_fy.size != 0 and self.bin_values_yfy.size != 0 and self.bin_values_ydy.size != 0:
            raise ValueError("Only one of bin_values_fy, bin_values_yfy, bin_values_ydy must be initialized (not three)")

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

        # check if sum of bin_values_f is positive
        if self.bin_values_fy.sum() <= 0:
            raise ValueError("Sum of bin_values_f must be positive")
        
        # # set normalized values
        logging.debug("self.fy.sum() is initialized to {}".format(self.fy.sum()))
        object.__setattr__(self, 'bin_values_fy_normalized', self.fy / self.fy.sum())
        logging.debug("bin_values_fy_normalized is initialized to {}".format(self.bin_values_fy_normalized))
        logging.debug("(self.y / self.yF) is initialized to {}".format((self.y / self.yF)))
        logging.debug("self.fy_norm is initialized to {}".format((self.fy_norm)))
        object.__setattr__(self, 'bin_values_dy_normalized', (self.y / self.yF) * self.bin_values_fy_normalized)
        object.__setattr__(self, 'bin_values_yfy_normalized', self.y * self.fy_norm)
        object.__setattr__(self, 'bin_values_ydy_normalized', self.y * self.bin_values_dy_normalized)

        if len(self.bin_centers) != len(self.bin_values_fy) \
                or len(self.bin_centers) != len(self.bin_values_yfy) \
                or len(self.bin_centers) != len(self.bin_values_ydy):
            raise ValueError("All arrays must have the same size")
            
    @property
    def num_bins(self):
        return len(self.bin_centers)
    
    @property
    def f_sum(self):
        return self.bin_values_fy.sum()
    
    @property
    def yF(self):
        return first_moment(bin_centers=self.y, bin_values=self.fy)
    
    @property
    def yD(self):
        return 0
    
    @property
    def y(self):
        return self.bin_centers
    
    @property
    def fy(self):
        return self.bin_values_fy
    
    @property
    def dy(self):
        return (self.y / self.yF) * self.fy

    @property
    def yfy(self):
        return self.bin_values_yfy
    
    @property
    def ydy(self):
        return self.bin_values_ydy

    @property
    def fy_norm(self):
        return self.bin_values_fy_normalized

    @property
    def fy_norm(self):
        return self.bin_values_dy_normalized

    @property
    def yfy_norm(self):
        return self.bin_values_yfy_normalized
    
    @property
    def ydy_norm(self):
        return self.bin_values_ydy_normalized

    @classmethod
    def from_lists(cls, bin_centers_list : list, bin_values_list : list =[], bin_values_yfy_list: list=[], bin_values_ydy_list: list=[]):
        bin_centers = np.array(bin_centers_list)
        bin_values_fy = np.array(bin_values_list) if bin_values_list else np.empty(0)
        bin_values_yfy = np.array(bin_values_yfy_list) if bin_values_yfy_list else np.empty(0)
        bin_values_ydy = np.array(bin_values_ydy_list) if bin_values_ydy_list else np.empty(0)
        return cls(bin_centers = bin_centers, bin_values_fy=bin_values_fy, bin_values_yfy=bin_values_yfy, bin_values_ydy=bin_values_ydy)
