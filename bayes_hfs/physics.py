"""
physics.py
Hyperfine physics

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable

import numpy as np
import pytensor.tensor as pt

import astropy.constants as c

_H_K_B_K_MHz = (c.h / c.k_B).to("K MHz-1").value  # h/k_B
_C_KMS = c.c.to("km/s").value
_C_CM_MHZ = c.c.to("cm MHz").value


def gaussian(x: float, center: float, fwhm: float) -> float:
    """Evaluate a normalized Gaussian function

    Parameters
    ----------
    x : float
        Position at which to evaluate
    center : float
        Gaussian centroid
    fwhm : float
        Gaussian FWHM line width

    Returns
    -------
    float
        Gaussian evaluated at x
    """
    return pt.exp(-4.0 * np.log(2.0) * (x - center) ** 2.0 / fwhm**2.0) * pt.sqrt(
        4.0 * np.log(2.0) / (np.pi * fwhm**2.0)
    )


def lorentzian(x: float, center: float, fwhm: float) -> float:
    """Evaluate a normalized Lorentzian function

    Parameters
    ----------
    x : float
        Position at which to evaluate
    center : float
        Centroid
    fwhm : float
        FWHM

    Returns
    -------
    float
        Lorentzian evaluated at x
    """
    return fwhm / (2.0 * np.pi) / ((x - center) ** 2.0 + (fwhm / 2.0) ** 2.0)


def calc_frequency(freq: float, velocity: float) -> float:
    """Apply the Doppler equation to calculate the frequency in the same frame as the velocity.

    Parameters
    ----------
    freq : float
        Frequencies. Output has same units.
    velocity : float
        Velocity (km/s)

    Returns
    -------
    float
        Radio-defined Doppler frequency
    """
    return freq * (1.0 - velocity / _C_KMS)


def calc_fwhm_freq(freq: float, fwhm: float) -> float:
    """Calculate the FWHM line width in frequency units

    Parameters
    ----------
    freq : float
        Frequencies. Output has same units.
    fwhm : float
        FWHM line width (km/s) in velocity units (length C)

    Returns
    -------
    float
        FWHM line width (MHz) in frequency units (shape C x N)
    """
    return freq * fwhm / _C_KMS


def calc_stat_weight(g: float, E: float, Tex: float) -> float:
    """Evaluate the statistical weight for a given excitation temperature.

    Parameters
    ----------
    g : float
        Degeneracy
    E : float
        Energy relative to ground state / k_B (K)
    Tex : float
        Excitation temperature (K)

    Returns
    -------
    float
        statistical weight g * exp(-E/(k*Tex))
    """
    return g * pt.exp(-E / Tex)


def calc_boltz_factor(freq: float, Tex: float) -> float:
    """Evaluate the Boltzmann factor from a given excitation temperature.
    B = Nu*gl/(Nl*gu) = exp(-h*freq/(k*Tex))

    Parameters
    ----------
    freq : float
        Frequency (MHz)
    Tex : float
        Excitation temperature (K)

    Returns
    -------
    float
        Boltzmann factor
    """
    return pt.exp(-_H_K_B_K_MHz * freq / Tex)


def calc_Tex(freq: float, boltz_factor: float) -> float:
    """Evaluate the excitation temperature from a given Boltzmann factor.

    Parameters
    ----------
    freq : float
        Frequency (MHz)
    boltz_factor : float
        Boltzmann factor = exp(-h*freq/(k*Tex))

    Returns
    -------
    float
        Excitation temperature
    """
    return -_H_K_B_K_MHz * freq / pt.log(boltz_factor)


def calc_pseudo_voigt(
    freq_axis: Iterable[float],
    frequency: Iterable[float],
    fwhm: Iterable[float],
    fwhm_L: Iterable[float],
) -> Iterable[float]:
    """Evaluate a pseudo Voight profile in order to aid in posterior exploration
    of the parameter space. This parameterization includes a latent variable fwhm_L, which
    can be conditioned on zero to analyze the posterior. We also consider the spectral
    channelization. We do not perform a full boxcar convolution, rather
    we approximate the convolution by assuming an equivalent FWHM for the
    boxcar kernel of 4 ln(2) / pi * channel_width ~= 0.88 * channel_width

    Parameters
    ----------
    freq_axis : Iterable[float]
        Observed frequency axis (MHz length S)
    frequency : Iterable[float]
        Cloud center frequency (MHz length C x N)
    fwhm : Iterable[float]
        Cloud FWHM line widths (MHz length C x N)
    fwhm_L : Iterable[float]
        Latent pseudo-Voigt profile Lorentzian FWHM (MHz length C x N)

    Returns
    -------
    Iterable[float]
        Line profile (MHz-1; shape S x C x N)
    """
    channel_size = pt.abs(freq_axis[1] - freq_axis[0])
    channel_fwhm = 4.0 * np.log(2.0) * channel_size / np.pi
    fwhm_conv = pt.sqrt(fwhm**2.0 + channel_fwhm**2.0 + fwhm_L**2.0)
    fwhm_L_frac = fwhm_L / fwhm_conv
    eta = (
        1.36603 * fwhm_L_frac - 0.47719 * fwhm_L_frac**2.0 + 0.11116 * fwhm_L_frac**3.0
    )

    # gaussian component
    gauss_part = gaussian(
        freq_axis[:, None, None], frequency[None, :, :], fwhm_conv[None, :, :]
    )

    # lorentzian component
    lorentz_part = lorentzian(
        freq_axis[:, None, None], frequency[None, :, :], fwhm_conv[None, :, :]
    )

    # linear combination
    return eta * lorentz_part + (1.0 - eta) * gauss_part


def calc_optical_depth(
    freq: float,
    gl: float,
    gu: float,
    Nl: float,
    Nu: float,
    Aul: float,
    line_profile: float,
) -> float:
    """Evaluate the total optical depth assuming a homogeneous medium. This
    is the integral of the absorption coefficient (Condon & Ransom eq. 7.55)

    Parameters
    ----------
    freq: float
        Transition frequency (MHz)
    gl: float
        Lower state degeneracy
    gu : float
        Upper state degeneracy
    Nl : float
        Lower state column density (cm-2)
    Nu : float
        Upper state column density (cm-2)
    Aul : float
        Einstein A coefficient (s-1)
    line_profile : float
        Line profile (MHz-1). Pass 1.0 to get the integrated optical depth.

    Returns
    -------
    float
        Total optical depth
    """
    return (
        _C_CM_MHZ**2.0  # cm2 MHz2
        / (8.0 * np.pi * freq**2.0)  # MHz-2
        * (Nl * gu / gl - Nu)  # cm-2
        * (Aul / 1.0e6)  # MHz
        * line_profile  # MHz-1
    )


def calc_TR(freq: float, boltz_factor: float) -> float:
    """Evaluate the radiation temperature (AKA Rayleigh-Jeans equivalent temperature,
    AKA brightness temperature). Note that we do not assume the R-J limit here.

    Parameters
    ----------
    freq : float
        frequency (MHz)
    boltz_factor : float
        Boltzmann factor

    Returns
    -------
    float
        Radiation temperature (AKA brightness temperature, K)
    """
    denom_factor = pt.switch(
        pt.eq(boltz_factor, 1.0), 0.0, 1.0 / (1.0 / boltz_factor - 1.0)
    )
    return _H_K_B_K_MHz * freq * denom_factor


def predict_tau_spectra(
    mol_data: dict,
    freq_axis: Iterable[float],
    tau: Iterable[float],
    velocity: Iterable[float],
    fwhm: Iterable[float],
    fwhm_L: float,
) -> Iterable[float]:
    """Predict the optical depth spectra from model parameters.

    Parameters
    ----------
    mol_data : dict
        Dictionary of molecular data returned by utils.get_molecule_data
        mol_data['freq'] contains C-length array of transition frequencies
    freq_axis : Iterable[float]
        Observed frequency axis (MHz length S)
    tau : Iterable[float]
        Total optical depth of each transition (shape C x N)
    velocity : Iterable[float]
        Velocity (km s-1) (length N)
    fwhm : Iterable[float]
        FWHM line width (km s-1) (length N)
    fwhm_L : float
        Latent pseudo-Voigt profile Lorentzian FWHM (km s-1)

    Returns
    -------
    Iterable[float]
        Predicted optical depth spectra (shape S x C x N)
    """
    # Frequency (MHz; shape: transitions, clouds)
    frequency = calc_frequency(mol_data["freq"][:, None], velocity[None, :])

    # Total FWHM line width in frequency units (MHz; shape: transitions, clouds)
    fwhm_freq = calc_fwhm_freq(mol_data["freq"][:, None], fwhm[None, :])

    # Latent Lorentzian FWHM line width in frequency units (MHz; shape: transitions, clouds)
    fwhm_L_freq = calc_fwhm_freq(mol_data["freq"][:, None], fwhm_L[None, :])

    # Line profile (MHz-1; shape: spectral, transitions, clouds)
    line_profile = calc_pseudo_voigt(freq_axis, frequency, fwhm_freq, fwhm_L_freq)

    # Optical depth  (shape: spectral, transitions, clouds)
    return tau[None, :, :] * line_profile


def radiative_transfer(
    freq_axis: Iterable[float],
    tau: Iterable[float],
    TR: Iterable[float],
    bg_temp: float,
) -> Iterable[float]:
    """Evaluate the radiative transfer to predict the emission spectrum. The emission
    spectrum is ON - OFF, where ON includes the attenuated emission of the background and
    the clouds, and the OFF is the emission of the background. Order of N clouds is
    assumed to be [nearest, ..., farthest]. The emission spectum is the Rayleigh-Jeans
    equivalent temperature, AKA the brightness temperature.

    Parameters
    ----------
    freq_axis : float
        Frequency axis (MHz) (length S)
    tau : Iterable[float]
        Optical depth spectra (shape S x C x N)
    TR : Iterable[float]
        Radiation temperature; shape: C x N)
    bg_temp : float
        Assumed background temperature

    Returns
    -------
    Iterable[float]
        Predicted emission Rayleigh-Jeans equivalent temperature
        (AKA brightness temperature) spectrum (K) (length S)
    """
    # nothing between us and the first cloud (shape S, 1)
    front_tau = pt.zeros_like(tau[:, 0, 0:1])

    # sum over transitions (shape S, N)
    sum_tau_cloud = tau.sum(axis=1)

    # cumulative sum over clouds, append front tau (shape S, N)
    # This is the total optical depth between us and cloud N
    # [0, tau(N=0), tau(N=0)+tau(N=1), ..., sum(tau)]
    sum_tau = pt.concatenate([front_tau, pt.cumsum(sum_tau_cloud, axis=1)], axis=1)
    total_tau = sum_tau[:, -1]

    # assume background radiation is in thermodynamic equilibrium at bg_temp
    # This is the R-J equivalent brightness temperature of the background (shape S)
    TR_bg = _H_K_B_K_MHz * freq_axis / (np.exp(_H_K_B_K_MHz * freq_axis / bg_temp) - 1)

    # Background is attenuated (or amplified) by all foreground clouds (shape S)
    TR_bg_attenuated = TR_bg * pt.exp(-total_tau)

    # Emission of each cloud, summed over transitions (shape S x N)
    TR_clouds = pt.sum(TR[None, :, :] * (1.0 - pt.exp(-tau)), axis=1)

    # Attenuation by foreground clouds (shape S x N)
    # [TB(N=0), TB(N=1)*exp(-tau(N=0)), TB(N=2)*exp(-tau(N=0)-tau(N=1)), ...]
    TR_clouds_attenuated = TR_clouds * pt.exp(-sum_tau[:, :-1])

    # Emission spectrum (shape S)
    TR_on = TR_bg_attenuated + pt.sum(TR_clouds_attenuated, axis=1)

    # ON - OFF
    return TR_on - TR_bg
