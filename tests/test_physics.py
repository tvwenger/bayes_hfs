"""
test_physics.py
tests for physics.py

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pickle
import numpy as np

from bayes_hfs import physics
from bayes_hfs import utils

try:
    _MOL_DATA, _MOL_METADATA = utils.get_molecule_data("C-13-N", fmin=100.0, fmax=200.0)
except:
    with open("docs/source/notebooks/mol_data_13CN.pkl", "rb") as f:
        _MOL_DATA = pickle.load(f)
    with open("docs/source/notebooks/mol_metadata_13CN.pkl", "rb") as f:
        _MOL_METADATA = pickle.load(f)
_MOL_DATA["GLO"] = 2 * _MOL_DATA["F1l"] + 1
_MOL_DATA = utils.supplement_molecule_data(_MOL_DATA, _MOL_METADATA)


def test_gaussian():
    x = np.linspace(-10.0, 10.0, 101)
    y = physics.gaussian(x, 0.0, 1.0).eval()
    assert not np.any(np.isnan(y))


def test_lorentzian():
    x = np.linspace(-10.0, 10.0, 101)
    y = physics.lorentzian(x, 0.0, 1.0)
    assert not np.any(np.isnan(y))


def test_calc_frequency():
    velocity = np.linspace(-10.0, 10.0, 101)
    frequency = physics.calc_frequency(_MOL_DATA["freq"][:, None], velocity)
    assert not np.any(np.isnan(frequency))


def test_calc_fwhm_freq():
    fwhm = np.array([10.0, 20.0, 30.0])
    fwhm_freq = physics.calc_fwhm_freq(_MOL_DATA["freq"][:, None], fwhm)
    assert not np.any(np.isnan(fwhm_freq))


def test_calc_stat_weight():
    g = 6.0
    E = 5.0
    Tex = 2.0
    stat_weight = physics.calc_stat_weight(g, E, Tex).eval()
    assert not np.isnan(stat_weight)


def test_calc_boltz_factor():
    freq = 112000.0
    Tex = 5.0
    boltz_factor = physics.calc_boltz_factor(freq, Tex).eval()
    assert not np.isnan(boltz_factor)


def test_calc_Tex():
    freq = 112000.0
    boltz_factor = 0.5
    Tex = physics.calc_Tex(freq, boltz_factor).eval()
    assert not np.isnan(Tex)


def test_calc_psuedo_voight():
    freq_axis = np.linspace(1000.0, 1100.0, 101)
    frequency = np.array([[1050.0, 1051.0], [1050.0, 1051.0]])
    fwhm = np.array([[10.0, 20.0], [10.0, 20.0]])
    fwhm_L = np.array([1.0, 1.0])
    line_profile = physics.calc_pseudo_voigt(freq_axis, frequency, fwhm, fwhm_L).eval()
    assert line_profile.shape == (101, 2, 2)


def test_calc_optical_depth():
    freq = 112000.0
    gu = 6.0
    gl = 2.0
    Nu = 5.0e11
    Nl = 1.0e12
    line_profile = 1.0
    Aul = 1.0e-12
    optical_depth = physics.calc_optical_depth(freq, gl, gu, Nl, Nu, Aul, line_profile)
    assert not np.isnan(optical_depth)


def test_calc_TR():
    freq = 112000.0
    boltz_factor = 0.5
    TR = physics.calc_TR(freq, boltz_factor).eval()
    assert not np.isnan(TR)


def test_predict_tau_spectra():
    freq_axis = np.linspace(113470.0, 113530.0, 500)
    tau = np.random.uniform(0.0, 1.0, size=(len(_MOL_DATA["freq"]), 2))
    velocity = np.array([-5.0, 5.0])
    fwhm = np.array([3.0, 5.0])
    fwhm_L = np.array([1.0, 1.0])
    tau_spectra = physics.predict_tau_spectra(
        _MOL_DATA, freq_axis, tau, velocity, fwhm, fwhm_L
    ).eval()
    assert tau_spectra.shape == (500, len(_MOL_DATA["freq"]), 2)
    assert not np.any(np.isnan(tau_spectra))


def test_radiative_transfer():
    freq_axis = np.linspace(113470.0, 113530.0, 500)
    tau = np.random.uniform(0.0, 1.0, size=(len(_MOL_DATA["freq"]), 2))
    velocity = np.array([-5.0, 5.0])
    fwhm = np.array([3.0, 5.0])
    fwhm_L = np.array([1.0, 1.0])
    tau_spectra = physics.predict_tau_spectra(
        _MOL_DATA, freq_axis, tau, velocity, fwhm, fwhm_L
    ).eval()
    freq = 113500.0
    boltz_factor = np.random.uniform(0.0, 1.0, size=(len(_MOL_DATA["freq"]), 2))
    TR = physics.calc_TR(freq, boltz_factor)
    bg_temp = 2.7
    brightness = physics.radiative_transfer(freq_axis, tau_spectra, TR, bg_temp).eval()
    assert brightness.shape == (500,)
    assert not np.any(np.isnan(brightness))
