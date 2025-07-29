"""
test_hfs_ratio_model.py
tests for HFSRatioModel

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pickle
import numpy as np

from bayes_spec import SpecData
from bayes_hfs import HFSRatioModel
from bayes_hfs import utils

try:
    _MOL_DATA_12CN, _MOL_METADATA_12CN = utils.get_molecule_data(
        "CN, v=0,1", fmin=100.0, fmax=200.0
    )
except:
    with open("docs/source/notebooks/mol_data_12CN.pkl", "rb") as f:
        _MOL_DATA_12CN = pickle.load(f)
    with open("docs/source/notebooks/mol_metadata_12CN.pkl", "rb") as f:
        _MOL_METADATA_12CN = pickle.load(f)
_MOL_DATA_12CN = _MOL_DATA_12CN[_MOL_DATA_12CN["Kl"] == 0]
_MOL_DATA_12CN["GLO"] = 2 * _MOL_DATA_12CN["F1l"]
_MOL_DATA_12CN = utils.supplement_molecule_data(_MOL_DATA_12CN, _MOL_METADATA_12CN)

try:
    _MOL_DATA_13CN, _MOL_METADATA_13CN = utils.get_molecule_data(
        "C-13-N", fmin=100.0, fmax=200.0
    )
except:
    with open("docs/source/notebooks/mol_data_13CN.pkl", "rb") as f:
        _MOL_DATA_13CN = pickle.load(f)
    with open("docs/source/notebooks/mol_metadata_13CN.pkl", "rb") as f:
        _MOL_METADATA_13CN = pickle.load(f)
_MOL_DATA_13CN["GLO"] = 2 * _MOL_DATA_13CN["F1l"] + 1
_MOL_DATA_13CN = utils.supplement_molecule_data(_MOL_DATA_13CN, _MOL_METADATA_13CN)


def test_hfs_ratio_model():
    freq_axis = np.linspace(113470.0, 113530.0, 1000)
    brightness = np.random.randn(1000)
    data = {
        "12CN": SpecData(freq_axis, brightness, 1.0),
        "13CN": SpecData(freq_axis, brightness, 1.0),
    }
    mol_keys = {
        "12CN": ["12CN"],
        "13CN": ["13CN"],
    }

    # Assume CTEX both
    model = HFSRatioModel(
        _MOL_DATA_12CN, _MOL_DATA_13CN, mol_keys, data, n_clouds=1, baseline_degree=1
    )
    model.add_priors()
    model.add_likelihood()
    assert model._validate()

    # Non-CTEX, Lorentzian line profile
    model = HFSRatioModel(
        _MOL_DATA_12CN, _MOL_DATA_13CN, mol_keys, data, n_clouds=1, baseline_degree=1
    )
    model.add_priors(assume_CTEX1=False, assume_CTEX2=False, prior_fwhm_L=1.0)
    model.add_likelihood()
    assert model._validate()
