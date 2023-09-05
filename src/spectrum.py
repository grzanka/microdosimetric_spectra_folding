from dataclasses import dataclass, field
from io import StringIO
import logging
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from src.helpers import (
    SpectrumBinningType,
    bin_edges,
    binning_type,
    first_moment,
    SpectrumValueType,
    others_from_dose_arrays,
    others_from_freq_arrays,
    others_from_x_times_dose_arrays,
    others_from_x_times_freq,
)
from src.checks import (
    check_if_array_holds_spectrum,
    check_if_bin_centers_valid,
    check_if_only_one_initialized,
    check_if_same_length_as_bin_centers,
)


@dataclass(frozen=True)
class SpectrumData:
    """Spectrum class. It is immutable. It can be initialized from bin_centers and one of bin_values_fy, bin_values_yfy, bin_values_ydy."""

    bin_centers: NDArray = field(default_factory=lambda: np.empty(0))

    bin_values_freq: NDArray = field(default_factory=lambda: np.empty(0))
    bin_values_freq_times_x: NDArray = field(default_factory=lambda: np.empty(0))
    bin_values_dose_times_x: NDArray = field(default_factory=lambda: np.empty(0))

    bin_values_dose: NDArray = field(default_factory=lambda: np.empty(0))

    bin_values_freq_normalized: NDArray = field(default_factory=lambda: np.empty(0))
    bin_values_freq_times_x_normalized: NDArray = field(
        default_factory=lambda: np.empty(0)
    )
    bin_values_dose_normalized: NDArray = field(default_factory=lambda: np.empty(0))
    bin_values_dose_times_x_normalized: NDArray = field(
        default_factory=lambda: np.empty(0)
    )

    binning_type: SpectrumBinningType = SpectrumBinningType.unknown

    bin_edges: NDArray = field(default_factory=lambda: np.empty(0))
    bin_widths: NDArray = field(default_factory=lambda: np.empty(0))

    bin_nums: int = 0
    norm: float = np.nan
    freq_mean: float = np.nan
    dose_mean: float = np.nan

    def __post_init__(self):
        check_if_bin_centers_valid(self.bin_centers)
        object.__setattr__(self, "num_bins", self.bin_centers.size)

        check_if_only_one_initialized(
            freq=self.bin_values_freq,
            freq_times_x=self.bin_values_freq_times_x,
            dose_times_x=self.bin_values_dose_times_x,
        )

        logging.debug(f"bin_values_freq: {self.bin_values_freq}")
        logging.debug(f"bin_values_freq_times_x: {self.bin_values_freq_times_x}")
        logging.debug(f"bin_values_dose_times_x: {self.bin_values_dose_times_x}")

        freq = dose = freq_times_x = dose_times_x = np.empty(0)
        if self.bin_values_freq.size != 0:
            freq = self.bin_values_freq
            freq_times_x, dose, dose_times_x = others_from_freq_arrays(
                x=self.bin_centers, freq=self.bin_values_freq
            )
        if self.bin_values_freq_times_x.size != 0:
            freq_times_x = self.bin_values_freq_times_x
            freq, dose, dose_times_x = others_from_x_times_freq(
                x=self.bin_centers, x_times_freq=freq_times_x
            )
        if self.bin_values_dose.size != 0:
            dose = self.bin_values_dose
            freq, freq_times_x, dose_times_x = others_from_dose_arrays(
                x=self.bin_centers, dose=self.bin_values_dose
            )
        if self.bin_values_dose_times_x.size != 0:
            dose_times_x = self.bin_values_dose_times_x
            freq, freq_times_x, dose = others_from_x_times_dose_arrays(
                x=self.bin_centers, x_times_dose=dose_times_x
            )
        if self.bin_values_freq.size == 0:
            object.__setattr__(self, "bin_values_freq", freq)
        if self.bin_values_freq_times_x.size == 0:
            object.__setattr__(self, "bin_values_freq_times_x", freq_times_x)
        object.__setattr__(self, "bin_values_dose", dose)
        object.__setattr__(self, "bin_values_dose_times_x", dose_times_x)

        check_if_same_length_as_bin_centers(
            bin_centers=self.bin_centers,
            fy=self.bin_values_freq,
            yfy=self.bin_values_freq_times_x,
            ydy=self.bin_values_dose_times_x,
        )

        # Set binning type
        object.__setattr__(self, "binning_type", binning_type(self.bin_centers))

        # Set bin edges and bin widths
        object.__setattr__(
            self, "bin_edges", bin_edges(self.bin_centers, self.binning_type)
        )
        object.__setattr__(self, "bin_widths", np.diff(self.bin_edges))

        # set means
        if self.bin_values_freq.size > 0:
            object.__setattr__(
                self,
                "freq_mean",
                first_moment(bin_edges=self.bin_edges, bin_values=self.bin_values_freq),
            )
        if self.bin_values_dose.size > 0:
            object.__setattr__(
                self,
                "dose_mean",
                first_moment(bin_edges=self.bin_edges, bin_values=self.bin_values_dose),
            )

        # set normalized values if bin_centers are initialized
        object.__setattr__(self, "norm", self.bin_values_freq @ self.bin_widths)
        freq_norm = (
            self.bin_values_freq / self.norm
            if self.norm != 0
            else np.zeros_like(self.bin_values_freq)
        )
        object.__setattr__(self, "bin_values_freq_normalized", freq_norm)
        freq_times_x_norm, dose_norm, dose_times_x_norm = others_from_freq_arrays(
            x=self.bin_centers, freq=self.bin_values_freq_normalized
        )
        object.__setattr__(
            self, "bin_values_freq_times_x_normalized", freq_times_x_norm
        )
        object.__setattr__(self, "bin_values_dose_normalized", dose_norm)
        object.__setattr__(
            self, "bin_values_dose_times_x_normalized", dose_times_x_norm
        )

    def bin_numbers(self, x: NDArray) -> NDArray:
        """Return the indices of the bins to which each value in input array belongs."""
        return np.digitize(x=x, bins=self.bin_edges) - 1

    def bin_number(self, x: float) -> float:
        """Return the index of the bins to which the value belongs."""
        return self.bin_numbers(np.array([x]))[0]

    def bin_values(
        self,
        x: NDArray,
        spectrum_value_type: SpectrumValueType = SpectrumValueType.freq_times_bin_centers,
    ) -> float:
        """Return the value of the bin to which the value belongs."""
        ind = self.bin_numbers(x)
        ind[ind < 0] = -1
        ind[ind >= self.bin_centers.size] = -1
        bin_values_extended = np.zeros(shape=(self.bin_centers.size + 2,))
        if spectrum_value_type == SpectrumValueType.freq:
            bin_values_extended[1:-1] = self.bin_values_freq
        elif spectrum_value_type == SpectrumValueType.freq_times_bin_centers:
            bin_values_extended[1:-1] = self.bin_values_freq_times_x
        elif spectrum_value_type == SpectrumValueType.dose_times_bin_centers:
            bin_values_extended[1:-1] = self.bin_values_dose_times_x
        return bin_values_extended.take(indices=ind + 1, mode="clip")

    def bin_value(
        self,
        x: float,
        spectrum_value_type: SpectrumValueType = SpectrumValueType.freq_times_bin_centers,
    ) -> float:
        """Return the value of the bin to which the value belongs."""
        return self.bin_values(np.array([x]), spectrum_value_type)[0]

    @classmethod
    def from_lists(
        cls,
        x: list = [],
        freq: list = [],
        freq_times_x: list = [],
        dose_times_x: list = [],
    ):
        freq_array = np.array(freq) if freq else np.empty(0)
        freq_times_x_array = np.array(freq_times_x) if freq_times_x else np.empty(0)
        dose_times_x_array = np.array(dose_times_x) if dose_times_x else np.empty(0)
        return cls(
            bin_centers=np.array(x),
            bin_values_freq=freq_array,
            bin_values_freq_times_x=freq_times_x_array,
            bin_values_dose_times_x=dose_times_x_array,
        )

    def __str__(self):
        fields = [
            (name, value)
            for name, value in self.__dict__.items()
            if isinstance(value, np.ndarray)
        ]

        output = ""
        for field_name, field_value in fields:
            output += f"{field_name}:\n{field_value}\n\n"

        return output


@dataclass(frozen=True)
class LinealEnergySpectrum:
    data: SpectrumData = field(default_factory=lambda: SpectrumData())

    @property
    def y(self) -> NDArray:
        return self.data.bin_centers

    @property
    def fy(self) -> NDArray:
        return self.data.bin_values_freq

    @property
    def yfy(self) -> NDArray:
        return self.data.bin_values_freq_times_x

    @property
    def ydy(self) -> NDArray:
        return self.data.bin_values_dose_times_x

    @property
    def yF(self) -> float:
        return self.data.freq_mean

    @property
    def yD(self) -> float:
        return self.data.dose_mean

    @property
    def norm(self) -> float:
        return self.data.norm

    @staticmethod
    def from_csv(
        file_path: Path, value_type: SpectrumValueType = SpectrumValueType.yfy, **kwargs
    ):
        """Load spectrum from csv file. The file must contain two columns: bin_centers and bin_values_fy."""
        data: SpectrumData = from_csv(file_path, value_type, **kwargs)
        return LinealEnergySpectrum(data=data)


@dataclass(frozen=True)
class SpecificEnergySpectrum:
    data: SpectrumData = field(default_factory=lambda: SpectrumData())
    site_diam_um: float = np.nan

    @property
    def z(self) -> NDArray:
        return self.data.bin_centers

    @property
    def fz(self) -> NDArray:
        return self.data.bin_values_freq

    @property
    def zfz(self) -> NDArray:
        return self.data.bin_values_freq_times_x

    @property
    def zdz(self) -> NDArray:
        return self.data.bin_values_dose_times_x

    @property
    def zF(self) -> float:
        return self.data.freq_mean

    @property
    def zD(self) -> float:
        return self.data.dose_mean

    @property
    def norm(self) -> float:
        return self.data.norm

    @staticmethod
    def from_csv(
        file_path: Path,
        site_diam_um: float,
        value_type: SpectrumValueType = SpectrumValueType.zfz,
        **kwargs,
    ):
        """Load spectrum from csv file. The file must contain two columns: bin_centers and bin_values."""
        data: SpectrumData = from_csv(file_path, value_type, **kwargs)
        return SpecificEnergySpectrum(data=data, site_diam_um=site_diam_um)


def from_array(
    data_array: NDArray,
    value_type: SpectrumValueType = SpectrumValueType.freq_times_bin_centers,
) -> SpectrumData:
    """Load spectrum from array. The array must contain two columns: bin_centers and bin_values_fy."""
    check_if_array_holds_spectrum(data_array)
    result = SpectrumData()
    if value_type == SpectrumValueType.freq:
        result = SpectrumData(
            bin_centers=data_array[:, 0], bin_values_freq=data_array[:, 1]
        )
    elif value_type == SpectrumValueType.freq_times_bin_centers:
        result = SpectrumData(
            bin_centers=data_array[:, 0], bin_values_freq_times_x=data_array[:, 1]
        )
    elif value_type == SpectrumValueType.dose_times_bin_centers:
        result = SpectrumData(
            bin_centers=data_array[:, 0], bin_values_dose_times_x=data_array[:, 1]
        )
    return result


def from_str(
    data_string: str,
    value_type: SpectrumValueType = SpectrumValueType.freq_times_bin_centers,
    **kwargs,
) -> SpectrumData:
    """Load spectrum from string. The string must contain two columns: bin_centers and bin_values_fy."""
    if data_string:
        data_array = np.genfromtxt(StringIO(data_string), **kwargs)
        check_if_array_holds_spectrum(data_array)
        result = from_array(data_array, value_type)
    else:
        result = SpectrumData()
    return result


def from_csv(
    file_path: Path,
    value_type: SpectrumValueType = SpectrumValueType.freq_times_bin_centers,
    **kwargs,
) -> SpectrumData:
    """Load spectrum from csv file. The file must contain two columns: bin_centers and bin_values_fy."""
    data_array = np.genfromtxt(file_path, **kwargs)
    check_if_array_holds_spectrum(data_array)
    result = from_array(data_array, value_type)
    return result


def specific_energy_spectum(
    lineal_energy_spectrum: LinealEnergySpectrum, site_diam_um: float
) -> SpecificEnergySpectrum:
    result = SpecificEnergySpectrum(
        # z = 0.204 * y / diam**2
        data=SpectrumData(
            bin_centers=0.204 * lineal_energy_spectrum.y / site_diam_um**2,
            bin_values_freq=lineal_energy_spectrum.fy,
        ),
        site_diam_um=site_diam_um,
    )
    return result


def lineal_energy_spectum(
    specific_energy_spectrum: SpecificEnergySpectrum,
) -> LinealEnergySpectrum:
    result = LinealEnergySpectrum(
        data=SpectrumData(
            bin_centers=(
                specific_energy_spectrum.z * specific_energy_spectrum.site_diam_um**2
            )
            / 0.204,
            bin_values_freq=specific_energy_spectrum.fz,
        )
    )
    return result
