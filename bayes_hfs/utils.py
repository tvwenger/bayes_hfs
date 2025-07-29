"""
utils.py
Hyperfine utilities

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np
from numpy.polynomial import Polynomial

import astropy.units as u
import astropy.constants as c
from astropy.table import Table
from astroquery.linelists.cdms import CDMS


def get_molecule_data(
    molecule: str,
    fmin: float = 0.0,
    fmax: float = 10000.0,
) -> dict:
    """Get molecular transition data from the JPL database. Lifted heavily from
    pyspeckit's get_molecular_parameters.

    Parameters
    ----------
    molecule : str
        Molecule name
    fmin : float, optional
        Minimum frequency (GHz), by default 0.0
    fmax : float, optional
        Maximum frequency (GHz), by default 10000.0

    Returns
    -------
    astropy.table.Table
        CDMS data
    astropy.table.Table
        CDMS metadata

    Raises
    ------
    ValueError
        Molecule not found in CDMS database
    """
    # get metadata
    species = CDMS.get_species_table()
    if molecule not in species["molecule"]:
        raise ValueError(f"{molecule} not found in database")
    mol_metadata = species[species["molecule"] == molecule][0]

    # get transition data
    mol_data = CDMS.query_lines(
        min_frequency=fmin * u.GHz,
        max_frequency=fmax * u.GHz,
        molecule=f"{mol_metadata['tag']:06d} {molecule}",
    )
    return mol_data, mol_metadata


def supplement_molecule_data(mol_data: Table, mol_metadata: Table) -> dict:
    """Supplement molecular transition data. Lifted heavily from
    pyspeckit's get_molecular_parameters. The user must first prune
    the table to only those transitions they wish to model, and they
    must manually add the lower state degeneracy (GLO) for each
    transition (e.g., mol_data['GLO'] = 2*mol_data['F1l'] + 1).

    Parameters
    ----------
    mol_data : astropy.table.Table
        Molecule data returned by get_molecule_data

    Returns
    -------
    dict
        Molecular transition data, with keys:
        "mol_weight" (float) : Molecular weight (Daltons)
        "freq" (Iterable[float]) : Rest frequencies (MHz)
        "Aul" (Iterable[float]) : Einstein A coefficients (s-1)
        "degu" (Iterable[float]) : Upper state degeneracies
        "Eu" (Iterable[float]) : Upper state energies (erg)
        "relative_int" (Iterable[float]) : Relative intensities
        "log10_Q_terms" (Iterable[float]) : Polynomial coefficients for logQ vs. logT (K)

    Raises
    ------
    ValueError
        Molecule not found in JPLSpec database
    """
    # partition function linear fit parameters
    log10_temps = np.log10(mol_metadata.meta["Temperature (K)"])
    log10_Q = np.array(
        [mol_metadata[key] for key in mol_metadata.keys() if "lg(Q" in key]
    )
    # drop nans
    bad = np.isnan(log10_Q)
    log10_temps = log10_temps[~bad]
    log10_Q = log10_Q[~bad]
    log10_Q_fit = Polynomial.fit(log10_temps, log10_Q, 1).convert()
    log10_Q_terms = log10_Q_fit.coef

    # rest frequencies, degeneracies, and upper energy levels
    freqs = mol_data["FREQ"].quantity
    freq_MHz = freqs.to(u.MHz).value
    degu = np.array(mol_data["GUP"])
    EL = mol_data["ELO"].quantity.to(u.erg, u.spectral())
    dE = freqs.to(u.erg, u.spectral())
    EU = EL + dE

    # need elower, eupper in inverse centimeter units
    elower_icm = mol_data["ELO"].quantity.to(u.cm**-1).value
    eupper_icm = elower_icm + (freqs.to(u.cm**-1, u.spectral()).value)

    # from Brett McGuire
    # https://github.com/bmcguir2/simulate_lte/blob/1f3f7c666946bc88c8d83c53389556a4c75c2bbd/simulate_lte.py#L2580-L2587

    # LGINT: Base 10 logarithm of the integrated intensity in units of nm2 MHz at 300 K.
    # (See Section 3 for conversions to other units.)
    # see also https://cdms.astro.uni-koeln.de/classic/predictions/description.html#description
    CT = 300.0
    logint = np.array(mol_data["LGINT"])  # this should just be a number
    # from CDMS website
    sijmu = (
        (
            np.exp(np.float64(-(elower_icm / 0.695) / CT))
            - np.exp(np.float64(-(eupper_icm / 0.695) / CT))
        )
        ** (-1)
        * ((10**logint) / freq_MHz)
        * (24025.120666)
        * 10.0 ** log10_Q_terms[0]
        * CT ** log10_Q_terms[1]
    )

    # aij formula from CDMS.  Verfied it matched spalatalogue's values
    aij = 1.16395e-20 * freq_MHz**3 * sijmu / degu

    # relative intensity
    relative_int = 10.0**logint
    relative_int = relative_int / relative_int.sum()

    # unique state ID based on quantum numbers
    QNL = [
        f"{row['Jl']} {row['Kl']} {row['vl']} {row['F1l']} {row['F2l']} {row['F3l']}"
        for row in mol_data
    ]
    QNU = [
        f"{row['Ju']} {row['Ku']} {row['vu']} {row['F1u']} {row['F2u']} {row['F3u']}"
        for row in mol_data
    ]

    # Get unique states and degeneracies
    states = np.array(QNL + QNU)
    degs = np.array(list(mol_data["GLO"]) + list(mol_data["GUP"]))

    # Energy/k_B (K)
    Es = np.array(
        [(E / c.k_B.to("erg K-1")).to("K").value for E in list(EL) + list(EU)]
    )

    # Keep unique
    unique_states, unique_idx = np.unique(states, return_index=True)
    unique_states = list(unique_states)
    unique_degs = degs[unique_idx]
    unique_Es = Es[unique_idx]

    states = {
        "state": unique_states,
        "deg": unique_degs,
        "E": unique_Es,
    }
    state_u_idx = [unique_states.index(Qu) for Qu in QNU]
    state_l_idx = [unique_states.index(Ql) for Ql in QNL]

    return {
        "mol_weight": mol_data["MOLWT"][0],  # molecular weight (Daltons)
        "freq": freqs.to(u.MHz).value,  # rest frequency (MHz)
        "Aul": aij,  # Einstein A (s-1)
        "relative_int": relative_int,  # relative intensity
        "states": states,
        "state_u_idx": state_u_idx,
        "state_l_idx": state_l_idx,
        "Gu": unique_degs[state_u_idx],
        "Gl": unique_degs[state_l_idx],
    }
