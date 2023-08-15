from dataclasses import dataclass, InitVar, field
import numpy as np

@dataclass(frozen=True)
class Spectrum:
    bin_centers: np.array = field(default_factory=lambda: np.empty(0))
    bin_values_f: np.array = field(default_factory=lambda: np.empty(0))
    bin_values_yfy: np.array = field(default_factory=lambda: np.empty(0))
    bin_values_ydy: np.array = field(default_factory=lambda: np.empty(0))
    def __post_init__(self):
        if self.bin_centers.size == 0:
            raise ValueError("bin_centers must be initialized")
        
        if self.bin_values_f.size == 0 and self.bin_values_yfy.size == 0 and self.bin_values_ydy.size == 0:
            raise ValueError("At least one of bin_values_f, bin_values_yfy, bin_values_ydy must be initialized (not zero)")
        if self.bin_values_f.size == 0 and self.bin_values_yfy.size != 0 and self.bin_values_ydy.size != 0 \
                or self.bin_values_f.size != 0 and self.bin_values_yfy.size == 0 and self.bin_values_ydy.size != 0 \
                or self.bin_values_f.size != 0 and self.bin_values_yfy.size != 0 and self.bin_values_ydy.size == 0:
            raise ValueError("Only one of bin_values_f, bin_values_yfy, bin_values_ydy must be initialized (not two)")
        if self.bin_values_f.size != 0 and self.bin_values_yfy.size != 0 and self.bin_values_ydy.size != 0:
            raise ValueError("Only one of bin_values_f, bin_values_yfy, bin_values_ydy must be initialized (not three)")

        if self.bin_values_f.size != 0:
            object.__setattr__(self, 'bin_values_yfy', self.bin_values_f * self.bin_centers)
            object.__setattr__(self, 'bin_values_ydy', self.bin_values_f * self.bin_centers**2)
        if self.bin_values_yfy.size != 0:
            object.__setattr__(self, 'bin_values_f', self.bin_values_yfy / self.bin_centers)
            object.__setattr__(self, 'bin_values_ydy', self.bin_values_yfy * self.bin_centers)
        if self.bin_values_ydy.size != 0:
            object.__setattr__(self, 'bin_values_f', self.bin_values_ydy / self.bin_centers**2)
            object.__setattr__(self, 'bin_values_yfy', self.bin_values_ydy / self.bin_centers)
        
        if len(self.bin_centers) != len(self.bin_values_f) \
                or len(self.bin_centers) != len(self.bin_values_yfy) \
                or len(self.bin_centers) != len(self.bin_values_ydy):
            raise ValueError("All arrays must have the same size")
            
    @property
    def num_bins(self):
        return len(self.bin_centers)

    @classmethod
    def from_lists(cls, bin_centers_list : list, bin_values_list : list =None, bin_values_yfy_list: list=None, bin_values_ydy_list: list=None):
        bin_centers = np.array(bin_centers_list)
        bin_values_f = np.array(bin_values_list) if bin_values_list is not None else np.empty(0)
        bin_values_yfy = np.array(bin_values_yfy_list) if bin_values_yfy_list is not None else np.empty(0)
        bin_values_ydy = np.array(bin_values_ydy_list) if bin_values_ydy_list is not None else np.empty(0)
        return cls(bin_centers, bin_values_f, bin_values_yfy, bin_values_ydy)
