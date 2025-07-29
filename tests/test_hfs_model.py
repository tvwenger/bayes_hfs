"""
test_hfs_model.py
Tests for HFSModel

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pickle
import numpy as np

from bayes_spec import SpecData
from bayes_hfs import HFSModel
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


def test_hfs_model():
    freq_axis = np.linspace(113470.0, 113530.0, 1000)
    brightness = np.random.randn(1000)
    data = {"observation": SpecData(freq_axis, brightness, 1.0)}

    # CTEX
    model = HFSModel(_MOL_DATA, data, n_clouds=1, baseline_degree=1)
    model.add_priors()
    model.add_likelihood()
    assert model._validate()

    # Non-CTEX, Lorentzian
    model = HFSModel(_MOL_DATA, data, n_clouds=1, baseline_degree=1)
    model.add_priors(assume_CTEX=False, prior_fwhm_L=1.0)
    model.add_likelihood()
    assert model._validate()
