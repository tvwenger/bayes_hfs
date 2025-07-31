"""
hfs_ratio_model.py
HFSRatioModel definition

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


class HFSRatioModel(BaseModel):
    """Definition of the HFSRatioModel."""

    def __init__(
        self,
        mol1_data: dict,
        mol2_data: dict,
        mol_keys: dict,
        *args,
        bg_temp: float = 2.7,
        Beff: float = 1.0,
        Feff: float = 1.0,
        **kwargs,
    ):
        """Initialize a new HFSRatioModel instance

        Parameters
        ----------
        mol1_data : dict
            Molecular data dictionary in the format returned by utils.supplement_molecule_data()
            for the first species. All transitions in the dictionary will be included in the model.
        mol2_data : dict
            Molecular data dictionary in the format returned by utils.supplement_molecule_data()
            for the second species. All transitions in the dictionary will be included in the model.
        mol_keys : dict
            Keys are the species names, values are iterables of the SpecData keys associated with each species.
            The order is important: the first key must be associated with mol1_data, and the second with mol2_data.
            e.g., {"12CN": ["12CN-1", "12CN-2"], "13CN": ["13CN-1", "13CN-2"]}
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
        self.mol1_data = mol1_data
        self.mol2_data = mol2_data
        self.mol_keys = mol_keys
        self.mol1 = list(self.mol_keys.keys())[0]
        self.mol2 = list(self.mol_keys.keys())[1]
        self.bg_temp = bg_temp
        self.Beff = Beff
        self.Feff = Feff

        # Add transitions and states to model
        coords = {
            f"transition_{self.mol1}": self.mol1_data["freq"],
            f"state_{self.mol1}": self.mol1_data["states"]["state"],
            f"transition_{self.mol2}": self.mol2_data["freq"],
            f"state_{self.mol2}": self.mol2_data["states"]["state"],
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
                f"log10_Ntot_{self.mol1}": r"$\log_{10} N_{\rm tot, "
                + f"{self.mol1}"
                + r"}$ (cm$^{-2}$)",
                f"log10_Ntot_{self.mol2}": r"$\log_{10} N_{\rm tot, "
                + f"{self.mol2}"
                + r"}$ (cm$^{-2}$)",
                "fwhm2": r"$\Delta V^2$ (km$^{2}$ s$^{-2}$)",
                "velocity": r"$v_{\rm LSR}$ (km s$^{-1}$)",
                "log10_Tex_CTEX": r"$\log_{10} T_{{\rm ex, CTEX}}$ (K)",
                f"Tex_{self.mol1}": r"$T_{\rm ex" + f"{self.mol1}" + r"}$ (K)",
                f"Tex_{self.mol2}": r"$T_{\rm ex" + f"{self.mol2}" + r"}$ (K)",
                "log10_CTEX_variance": r"$\log_{10} \sigma_{\rm CTEX}^2$",
                f"tau_{self.mol1}": r"$\tau_{\rm " + f"{self.mol1}" + r"$",
                f"tau_{self.mol2}": r"$\tau_{\rm " + f"{self.mol2}" + r"$",
                f"tau_total_{self.mol1}": r"$\tau_{\rm tot, " + f"{self.mol1}" + r"}$",
                f"tau_total_{self.mol2}": r"$\tau_{\rm tot, " + f"{self.mol2}" + r"}$",
                f"TR_{self.mol1}": r"$T_{R, \rm " + f"{self.mol1}" + r"$ (K)",
                f"TR_{self.mol2}": r"$T_{R, \rm " + f"{self.mol2}" + r"$ (K)",
                "ratio": r"$N_{\rm tot, "
                + f"{self.mol2}"
                + r"}/N_{\rm tot, "
                + f"{self.mol1}"
                + r"}$",
                "fwhm_L": r"$\Delta V_L$ (km s$^{-1}$)",
            }
        )

    def add_priors(
        self,
        prior_log10_Ntot1: Iterable[float] = [13.5, 0.5],
        prior_ratio: float = 0.1,
        prior_fwhm2: float = 1.0,
        prior_velocity: Iterable[float] = [-10.0, 10.0],
        prior_log10_Tex_CTEX: Iterable[float] = [0.75, 0.25],
        assume_CTEX1: bool = True,
        assume_CTEX2: bool = True,
        prior_log10_CTEX_variance: float = [-4.0, 1.0],
        clip_weights: Optional[float] = 1.0e-9,
        clip_tau: Optional[float] = -10.0,
        prior_fwhm_L: Optional[float] = None,
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_log10_Ntot1 : Iterable[float], optional
            Prior distribution on log10 total column density (cm-2) of the first species,
            by default [13.5, 0.5], where
            log10_Ntot1 ~ Normal(mu=prior[0], sigma=prior[1])
        prior_ratio : float, optional
            Prior distribution on the ratio Ntot2/Ntot1, by default 0.1, where
            ratio ~ HalfNormal(sigma=prior)
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
        assume_CTEX1 : bool, optional
            Assume that every transition of the first species has the same excitation temperature, by default True.
        assume_CTEX2 : bool, optional
            Assume that every transition of the second species has the same excitation temperature, by default True.
        prior_log10_CTEX_variance : Iterable[float], optional
            Prior distribution on the log10_variance of departures from CTEX, by default [-6.0, 1.0], where
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
            # total column density of the first species (cm-2; shape: clouds)
            log10_Ntot1_norm = pm.Normal(
                f"log10_Ntot_{self.mol1}_norm", mu=0.0, sigma=1.0, dims="cloud"
            )
            log10_Ntot1 = pm.Deterministic(
                f"log10_Ntot_{self.mol1}",
                prior_log10_Ntot1[0] + prior_log10_Ntot1[1] * log10_Ntot1_norm,
                dims="cloud",
            )
            Ntot1 = pt.power(10.0, log10_Ntot1)

            # column density ratio (shape: clouds)
            ratio_norm = pm.HalfNormal("ratio_norm", sigma=1.0, dims="cloud")
            ratio = pm.Deterministic("ratio", prior_ratio * ratio_norm, dims="cloud")

            # FWHM^2 (km2 s-2; shape: clouds)
            fwhm2_norm = pm.ChiSquared("fwhm2_norm", nu=1, dims="cloud")
            _ = pm.Deterministic("fwhm2", prior_fwhm2 * fwhm2_norm, dims="cloud")

            # Pseudo-Voigt profile latent variable (km/s)
            if prior_fwhm_L is not None:
                fwhm_L_norm = pm.HalfNormal("fwhm_L_norm", sigma=1.0, dims="cloud")
                _ = pm.Deterministic("fwhm_L", prior_fwhm_L * fwhm_L_norm, dims="cloud")
            else:
                _ = pm.Data("fwhm_L", np.zeros(self.n_clouds), dims="cloud")

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

            # Total column density of the second species (cm-2; shape: clouds)
            Ntot2 = ratio * Ntot1
            _ = pm.Deterministic(
                f"log10_Ntot_{self.mol2}", pt.log10(Ntot2), dims="cloud"
            )

            # CTEX variance (inverse concentration) (shape: clouds)
            if not (assume_CTEX1 and assume_CTEX2):
                log10_CTEX_variance_norm = pm.HalfNormal(
                    "log10_CTEX_variance_norm", sigma=1.0, dims="cloud"
                )
                log10_CTEX_variance = pm.Deterministic(
                    "log10_CTEX_variance",
                    prior_log10_CTEX_variance[0]
                    + prior_log10_CTEX_variance[1] * log10_CTEX_variance_norm,
                    dims="cloud",
                )

            # CTEX statistical weights (shape: clouds, states)
            CTEX_weights_mol1 = physics.calc_stat_weight(
                self.mol1_data["states"]["deg"][None, :],
                self.mol1_data["states"]["E"][None, :],
                10.0 ** log10_Tex_CTEX[:, None],
            )
            CTEX_weights_mol1 = pm.Deterministic(
                f"CTEX_weights_{self.mol1}",
                CTEX_weights_mol1,
                dims=["cloud", f"state_{self.mol1}"],
            )

            if assume_CTEX1:
                # Excitation temperature (K, shape: clouds)
                Tex_mol1 = pm.Deterministic(
                    f"Tex_{self.mol1}", 10.0**log10_Tex_CTEX, dims="cloud"
                )

                # Boltzmann factor (shape: transition, cloud)
                boltz_factor_mol1 = physics.calc_boltz_factor(
                    self.mol1_data["freq"][:, None], Tex_mol1[None, :]
                )

                # State column densities (cm-2; shape: clouds, states)
                N_state_mol1 = (
                    Ntot1[:, None]
                    * CTEX_weights_mol1
                    / pt.sum(CTEX_weights_mol1, axis=1, keepdims=True)
                )

                # Upper state column densities (cm-2; shape: transitions, clouds)
                Nu_mol1 = pt.stack(
                    [N_state_mol1[:, idx] for idx in self.mol1_data["state_u_idx"]]
                )
                Nl_mol1 = pt.stack(
                    [N_state_mol1[:, idx] for idx in self.mol1_data["state_l_idx"]]
                )
            else:
                # CTEX concentration (shape: clouds, state)
                CTEX_concentration_mol1 = (
                    CTEX_weights_mol1 / pt.power(10.0, log10_CTEX_variance)[:, None]
                )

                # Dirichlet state fraction (shape: cloud, state)
                weights_mol1_norm = pm.Dirichlet(
                    f"weights_{self.mol1}_norm",
                    a=CTEX_concentration_mol1,
                    dims=["cloud", f"state_{self.mol1}"],
                )
                # Prevent weights=0
                weights_mol1 = pt.clip(
                    weights_mol1_norm, clip_weights, 1.0 - clip_weights
                )

                # State column densities (cm-2; shape: clouds, states)
                N_state_mol1 = (
                    Ntot1[:, None]
                    * weights_mol1
                    / pt.sum(weights_mol1, axis=1, keepdims=True)
                )

                # Upper state column densities (cm-2; shape: transitions, clouds)
                Nu_mol1 = pt.stack(
                    [N_state_mol1[:, idx] for idx in self.mol1_data["state_u_idx"]]
                )
                Nl_mol1 = pt.stack(
                    [N_state_mol1[:, idx] for idx in self.mol1_data["state_l_idx"]]
                )

                # Boltzmann factor (shape: transition, cloud)
                boltz_factor_mol1 = (
                    Nu_mol1
                    * self.mol1_data["Gl"][:, None]
                    / (Nl_mol1 * self.mol1_data["Gu"][:, None])
                )

                # Excitation temperature (shape: transition, cloud)
                _ = pm.Deterministic(
                    f"Tex_{self.mol1}",
                    physics.calc_Tex(
                        self.mol1_data["freq"][:, None], boltz_factor_mol1
                    ),
                    dims=[f"transition_{self.mol1}", "cloud"],
                )

            # Optical depth (shape: transitions, clouds)
            tau_mol1 = pm.Deterministic(
                f"tau_{self.mol1}",
                pt.clip(
                    physics.calc_optical_depth(
                        self.mol1_data["freq"][:, None],
                        self.mol1_data["Gl"][:, None],
                        self.mol1_data["Gu"][:, None],
                        Nl_mol1,
                        Nu_mol1,
                        self.mol1_data["Aul"][:, None],
                        1.0,  # integrated line profile
                    ),
                    clip_tau,
                    pt.inf,
                ),
                dims=[f"transition_{self.mol1}", "cloud"],
            )

            # Total optical depth (shape: clouds)
            _ = pm.Deterministic(
                f"tau_total_{self.mol1}", pt.sum(tau_mol1, axis=0), dims="cloud"
            )

            # Radiation temperature (K; shape: transitions, clouds)
            # catch masers
            _ = pm.Deterministic(
                f"TR_{self.mol1}",
                physics.calc_TR(self.mol1_data["freq"][:, None], boltz_factor_mol1),
                dims=[f"transition_{self.mol1}", "cloud"],
            )

            # CTEX statistical weights (shape: clouds, states)
            CTEX_weights_mol2 = physics.calc_stat_weight(
                self.mol2_data["states"]["deg"][None, :],
                self.mol2_data["states"]["E"][None, :],
                10.0 ** log10_Tex_CTEX[:, None],
            )
            CTEX_weights_mol2 = pm.Deterministic(
                f"CTEX_weights_{self.mol2}",
                CTEX_weights_mol2,
                dims=["cloud", f"state_{self.mol2}"],
            )

            if assume_CTEX2:
                # Excitation temperature (K, shape: clouds
                Tex_mol2 = pm.Deterministic(
                    f"Tex_{self.mol2}", 10.0**log10_Tex_CTEX, dims="cloud"
                )

                # Boltzmann factor (shape: transition, cloud)
                boltz_factor_mol2 = physics.calc_boltz_factor(
                    self.mol2_data["freq"][:, None], Tex_mol2[None, :]
                )

                # State column densities (cm-2; shape: clouds, states)
                N_state_mol2 = (
                    Ntot2[:, None]
                    * CTEX_weights_mol2
                    / pt.sum(CTEX_weights_mol2, axis=1, keepdims=True)
                )

                # Upper state column densities (cm-2; shape: transitions, clouds)
                Nu_mol2 = pt.stack(
                    [N_state_mol2[:, idx] for idx in self.mol2_data["state_u_idx"]]
                )
                Nl_mol2 = pt.stack(
                    [N_state_mol2[:, idx] for idx in self.mol2_data["state_l_idx"]]
                )
            else:
                # CTEX concentration (shape: clouds, state)
                CTEX_concentration_mol2 = (
                    CTEX_weights_mol2 / pt.power(10.0, log10_CTEX_variance)[:, None]
                )

                # Dirichlet state fraction (shape: cloud, state)
                weights_mol2_norm = pm.Dirichlet(
                    f"weights_{self.mol2}_norm",
                    a=CTEX_concentration_mol2,
                    dims=["cloud", f"state_{self.mol2}"],
                )
                # Prevent weights=0
                weights_mol2 = pt.clip(
                    weights_mol2_norm, clip_weights, 1.0 - clip_weights
                )

                # State column densities (cm-2; shape: clouds, states)
                N_state_mol2 = (
                    Ntot2[:, None]
                    * weights_mol2
                    / pt.sum(weights_mol2, axis=1, keepdims=True)
                )

                # Upper state column densities (cm-2; shape: transitions, clouds)
                Nu_mol2 = pt.stack(
                    [N_state_mol2[:, idx] for idx in self.mol2_data["state_u_idx"]]
                )
                Nl_mol2 = pt.stack(
                    [N_state_mol2[:, idx] for idx in self.mol2_data["state_l_idx"]]
                )

                # Boltzmann factor (shape: transition, cloud)
                boltz_factor_mol2 = (
                    Nu_mol2
                    * self.mol2_data["Gl"][:, None]
                    / (Nl_mol2 * self.mol2_data["Gu"][:, None])
                )

                # Excitation temperature (shape: transition, cloud)
                _ = pm.Deterministic(
                    f"Tex_{self.mol2}",
                    physics.calc_Tex(
                        self.mol2_data["freq"][:, None], boltz_factor_mol2
                    ),
                    dims=[f"transition_{self.mol2}", "cloud"],
                )

            # Optical depth (shape: transitions, clouds)
            tau_mol2 = pm.Deterministic(
                f"tau_{self.mol2}",
                pt.clip(
                    physics.calc_optical_depth(
                        self.mol2_data["freq"][:, None],
                        self.mol2_data["Gl"][:, None],
                        self.mol2_data["Gu"][:, None],
                        Nl_mol2,
                        Nu_mol2,
                        self.mol2_data["Aul"][:, None],
                        1.0,  # integrated line profile
                    ),
                    clip_tau,
                    pt.inf,
                ),
                dims=[f"transition_{self.mol2}", "cloud"],
            )

            # Total optical depth (shape: clouds)
            _ = pm.Deterministic(
                f"tau_total_{self.mol2}", pt.sum(tau_mol2, axis=0), dims="cloud"
            )

            # Radiation temperature (K; shape: transitions, clouds)
            # catch masers
            _ = pm.Deterministic(
                f"TR_{self.mol2}",
                physics.calc_TR(self.mol2_data["freq"][:, None], boltz_factor_mol2),
                dims=[f"transition_{self.mol2}", "cloud"],
            )

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "observation"."""
        # Predict baseline models
        baseline_models = self.predict_baseline()

        # Predict all spectra
        for label, dataset in self.data.items():
            if label in self.mol_keys[self.mol1]:
                mol_data = self.mol1_data
                molecule = self.mol1
            elif label in self.mol_keys[self.mol2]:
                mol_data = self.mol2_data
                molecule = self.mol2
            else:  # pragma: no cover
                raise ValueError(f"Invalid dataset label: {label}")

            # Optical depth spectra (shape: spectral, transitions, clouds)
            tau_spectra = physics.predict_tau_spectra(
                mol_data,
                dataset.spectral,
                self.model[f"tau_{molecule}"],
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
                    self.model[f"TR_{molecule}"],
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
