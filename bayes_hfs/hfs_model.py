"""
hfs_model.py
HFSModel definition

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable, Optional

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bayes_spec import BaseModel

from bayes_hfs import physics


class HFSModel(BaseModel):
    """Definition of the HFSModel."""

    def __init__(
        self,
        mol_data: dict,
        *args,
        bg_temp: float = 2.7,
        Beff: float = 1.0,
        Feff: float = 1.0,
        **kwargs,
    ):
        """Initialize a new HFSModel instance

        Parameters
        ----------
        mol_data : dict
            Molecular data dictionary in the format returned by utils.supplement_molecule_data().
            All transitions in the dictionary will be included in the model.
        bg_temp : float, optional
            Assumed background temperature (K), by default 2.7
        Beff : float, optional
            Beam efficiency, by default 1.0
        Feff : float, optional
            Forward efficiency, by default 1.0
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Save inputs
        self.mol_data = mol_data
        self.bg_temp = bg_temp
        self.Beff = Beff
        self.Feff = Feff

        # Add transitions and states to model
        coords = {
            "transition": self.mol_data["freq"],
            "state": self.mol_data["states"]["state"],
        }
        self.model.add_coords(coords=coords)

        # Select features used for posterior clustering
        self._cluster_features += [
            "fwhm2",
            "velocity",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "log10_Ntot": r"$\log_{10} N_{\rm tot}$ (cm$^{-2}$)",
                "fwhm2": r"$\Delta V^2$ (km$^{2}$ s$^{-2}$)",
                "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
                "log10_Tex_CTEX": r"$\log_{10} T_{\rm ex, CTEX}$ (K)",
                "Tex": r"$T_{\rm ex}$ (K)",
                "log10_CTEX_variance": r"$\log_{10} \sigma_{\rm CTEX}^2$",
                "tau": r"$\tau$",
                "tau_total": r"$\tau_{\rm tot}$",
                "TR": r"$T_R$ (K)",
                "fwhm_L": r"$\Delta V_L$ (km s$^{-1}$)",
            }
        )

    def add_priors(
        self,
        prior_log10_Ntot: Iterable[float] = [13.5, 0.25],
        prior_fwhm2: float = 1.0,
        prior_velocity: Iterable[float] = [-10.0, 10.0],
        prior_log10_Tex_CTEX: Iterable[float] = [0.75, 0.1],
        assume_CTEX: bool = True,
        prior_log10_CTEX_variance: float = [-4.0, 1.0],
        clip_weights: Optional[float] = 1.0e-9,
        clip_tau: Optional[float] = -10.0,
        prior_fwhm_L: Optional[float] = None,
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_log10_Ntot : Iterable[float], optional
            Prior distribution on log10 total column density (cm-2),
            by default [13.5, 0.5], where
            log10_Ntot ~ Normal(mu=prior[0], sigma=prior[1])
        prior_fwhm2 : float, optional
            Prior distribution on FWHM^2  (km2 s-2), by default 1.0, where
            fwhm2 ~ prior * ChiSquared(nu=1)
            i.e., half-normal on FWHM
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [-10.0, 10.0], where
            velocity_norm ~ Beta(alpha=2.0, beta=2.0)
            velocity ~ prior[0] + (prior[1] - prior[0]) * velocity_norm
        prior_log10_Tex_CTEX : Iterable[float], optional
            Prior distribution on log10 CTEX excitation temperature (K), by default [0.75, 0.25], where
            log10_Tex_CTEX ~ Normal(mu=prior[0], sigma=prior[1])
        assume_CTEX : bool, optional
            Assume that every transition has the same excitation temperature, by default True.
        prior_log10_CTEX_variance : Iterable[float], optional
            Prior distribution on the log10_variance of departures from CTEX, by default [-4.0, 1.0], where
            log10_CTEX_variance ~ prior[0] + HalfNormal(sigma=prior[1])
            stat_weights ~ Dirichlet(a=len(states)*CTEX_stat_weights/CTEX_variance)
            where CTEX_stat_weights are derived from Tex
        clip_weights : Optional[float], optional
            Clip weights between [clip, 1.0-clip], by default 1.0e-9
        clip_tau : Optional[float], optional
            Clip masers by truncating optical depths below this value, by default -10.0
        prior_fwhm_L : Optional[float], optional
            Prior distribution on the pseudo-Voight Lorentzian profile line width (km/s),
            by default None, where
            fwhm_L ~ HalfNormal(sigma=prior_fwhm_L)
            If None, the line profile is assumed Gaussian.
        prior_baseline_coeffs : Optional[dict[str, Iterable[float]]], optional
            Prior distribution on the normalized baseline polynomial coefficients, by default None.
            Keys are dataset names and values are lists of length `baseline_degree+1`. If None, use
            `[1.0]*(baseline_degree+1)` for each dataset.
        """
        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # total column density (cm-2; shape: clouds)
            log10_Ntot_norm = pm.Normal(
                "log10_Ntot_norm", mu=0.0, sigma=1.0, dims="cloud"
            )
            log10_Ntot = pm.Deterministic(
                "log10_Ntot",
                prior_log10_Ntot[0] + prior_log10_Ntot[1] * log10_Ntot_norm,
                dims="cloud",
            )
            Ntot = pt.power(10.0, log10_Ntot)

            # FWHM^2 (km2 s-2; shape: clouds)
            fwhm2_norm = pm.ChiSquared("fwhm2_norm", nu=1, dims="cloud")
            _ = pm.Deterministic("fwhm2", prior_fwhm2 * fwhm2_norm, dims="cloud")

            # Velocity (km/s; shape: clouds)
            velocity_norm = pm.Beta("velocity_norm", alpha=2.0, beta=2.0, dims="cloud")
            _ = pm.Deterministic(
                "velocity",
                prior_velocity[0]
                + (prior_velocity[1] - prior_velocity[0]) * velocity_norm,
                dims="cloud",
            )

            # Upper->lower excitation temperature (K; shape: clouds)
            log10_Tex_CTEX_norm = pm.Normal(
                "log10_Tex_CTEX_norm", mu=0.0, sigma=1.0, dims="cloud"
            )
            log10_Tex_CTEX = pm.Deterministic(
                "log10_Tex_CTEX",
                prior_log10_Tex_CTEX[0] + prior_log10_Tex_CTEX[1] * log10_Tex_CTEX_norm,
                dims="cloud",
            )

            # CTEX statistical weights (shape: clouds, states)
            CTEX_weights = physics.calc_stat_weight(
                self.mol_data["states"]["deg"][None, :],
                self.mol_data["states"]["E"][None, :],
                10.0 ** log10_Tex_CTEX[:, None],
            )
            CTEX_weights = pm.Deterministic(
                "CTEX_weights",
                CTEX_weights / pt.sum(CTEX_weights, axis=1)[:, None],
                dims=["cloud", "state"],
            )

            if assume_CTEX:
                # constant across transitions (K; shape: clouds)
                Tex = pm.Deterministic("Tex", 10.0**log10_Tex_CTEX, dims="cloud")

                # Boltzmann factor (shape: transition, cloud)
                boltz_factor = physics.calc_boltz_factor(
                    self.mol_data["freq"][:, None], Tex[None, :]
                )

                # State column densities (cm-2; shape: clouds, states)
                N_state = Ntot[:, None] * CTEX_weights

                # Upper state column densities (cm-2; shape: transitions, clouds)
                Nu = pt.stack([N_state[:, idx] for idx in self.mol_data["state_u_idx"]])
                Nl = pt.stack([N_state[:, idx] for idx in self.mol_data["state_l_idx"]])
            else:
                # CTEX variance (inverse concentration) (shape: clouds)
                log10_CTEX_variance_norm = pm.HalfNormal(
                    "log10_CTEX_variance_norm", sigma=1.0, dims="cloud"
                )
                log10_CTEX_variance = pm.Deterministic(
                    "log10_CTEX_variance",
                    prior_log10_CTEX_variance[0]
                    + prior_log10_CTEX_variance[1] * log10_CTEX_variance_norm,
                    dims="cloud",
                )

                # CTEX concentration (shape: clouds, state)
                CTEX_concentration = (
                    len(self.model.coords["state"])
                    * CTEX_weights
                    / pt.power(10.0, log10_CTEX_variance)[:, None]
                )

                # Dirichlet state fraction (shape: cloud, state)
                weights = pm.Dirichlet(
                    "weights",
                    a=CTEX_concentration,
                    dims=["cloud", "state"],
                )
                # Prevent weights=0
                weights = pt.clip(weights, clip_weights, 1.0 - clip_weights)
                weights = weights / pt.sum(weights, axis=1)[:, None]

                # State column densities (cm-2; shape: clouds, states)
                N_state = Ntot[:, None] * weights

                # Upper state column densities (cm-2; shape: transitions, clouds)
                Nu = pt.stack([N_state[:, idx] for idx in self.mol_data["state_u_idx"]])
                Nl = pt.stack([N_state[:, idx] for idx in self.mol_data["state_l_idx"]])

                # Boltzmann factor (shape: transition, cloud)
                boltz_factor = (
                    Nu
                    * self.mol_data["Gl"][:, None]
                    / (Nl * self.mol_data["Gu"][:, None])
                )

                # Excitation temperature (shape: transition, cloud)
                Tex = pm.Deterministic(
                    "Tex",
                    physics.calc_Tex(self.mol_data["freq"][:, None], boltz_factor),
                    dims=["transition", "cloud"],
                )

            # Optical depth (shape: transitions, clouds)
            tau = pm.Deterministic(
                "tau",
                pt.clip(
                    physics.calc_optical_depth(
                        self.mol_data["freq"][:, None],
                        self.mol_data["Gl"][:, None],
                        self.mol_data["Gu"][:, None],
                        Nl,
                        Nu,
                        self.mol_data["Aul"][:, None],
                        1.0,  # integrated line profile
                    ),
                    clip_tau,
                    pt.inf,
                ),
                dims=["transition", "cloud"],
            )

            # Total optical depth (shape: clouds)
            _ = pm.Deterministic("tau_total", pt.sum(tau, axis=0), dims="cloud")

            # Radiation temperature (K; shape: transitions, clouds)
            _ = pm.Deterministic(
                "TR",
                physics.calc_TR(self.mol_data["freq"][:, None], boltz_factor),
                dims=["transition", "cloud"],
            )

            # Pseudo-Voigt profile latent variable (km/s)
            if prior_fwhm_L is not None:
                fwhm_L_norm = pm.HalfNormal("fwhm_L_norm", sigma=1.0, dims="cloud")
                _ = pm.Deterministic("fwhm_L", prior_fwhm_L * fwhm_L_norm, dims="cloud")
            else:
                _ = pm.Data("fwhm_L", np.zeros(self.n_clouds), dims="cloud")

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "observation"."""
        # Predict baseline models
        baseline_models = self.predict_baseline()

        # Predict all spectra
        for label, dataset in self.data.items():
            # Optical depth spectra (shape: spectral, transitions, clouds)
            tau_spectra = physics.predict_tau_spectra(
                self.mol_data,
                dataset.spectral,
                self.model["tau"],
                self.model["velocity"],
                pt.sqrt(self.model["fwhm2"]),
                self.model["fwhm_L"],
            )

            # Radiative transfer (shape: spectral)
            predicted_line = (
                self.Beff
                / self.Feff
                * physics.radiative_transfer(
                    dataset.spectral,
                    tau_spectra,
                    self.model["TR"],
                    self.bg_temp,
                )
            )

            # Add baseline model
            predicted = predicted_line + baseline_models[label]

            with self.model:
                # Evaluate likelihood
                _ = pm.Normal(
                    label,
                    mu=predicted,
                    sigma=dataset.noise,
                    observed=dataset.brightness,
                )
