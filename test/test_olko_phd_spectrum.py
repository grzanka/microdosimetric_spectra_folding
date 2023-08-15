import numpy as np
import pytest
from src.spectrum import Spectrum

def test_y(spectrum_fig3p3_olko_phd):
    assert np.array_equal(spectrum_fig3p3_olko_phd.y, [1,2,3])
    assert spectrum_fig3p3_olko_phd.y.shape == (3,)
    assert spectrum_fig3p3_olko_phd.y.ndim == 1


def test_fy(spectrum_fig3p3_olko_phd):
    assert np.array_equal(spectrum_fig3p3_olko_phd.fy, [2,2,2])
    assert np.unique(spectrum_fig3p3_olko_phd.fy) == pytest.approx(2)
    assert spectrum_fig3p3_olko_phd.fy.shape == (3,)
    assert spectrum_fig3p3_olko_phd.fy.ndim == 1
    assert np.unique(spectrum_fig3p3_olko_phd.fy_norm) == pytest.approx(1/3)

def test_yfy(spectrum_fig3p3_olko_phd):
    assert np.array_equal(spectrum_fig3p3_olko_phd.yfy, [2,4,6])
    assert np.array_equal(spectrum_fig3p3_olko_phd.yfy_norm, [1/3,2/3,1])

def test_dy(spectrum_fig3p3_olko_phd):
    assert np.array_equal(spectrum_fig3p3_olko_phd.dy, [1,2,3])

def test_dy(spectrum_fig3p3_olko_phd):
    assert np.array_equal(spectrum_fig3p3_olko_phd.ydy, [1,4,9])


