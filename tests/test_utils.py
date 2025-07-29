"""
test_utils.py
tests for utils.py

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pytest
import pickle
from bayes_hfs import utils


def test_get_molecule_data():
    with pytest.raises(ValueError):
        utils.get_molecule_data(molecule="BAD MOLECULE NAME")

    try:
        data, metadata = utils.get_molecule_data("C-13-N", fmin=100.0, fmax=200.0)
    except:
        with open("docs/source/notebooks/mol_data_13CN.pkl", "rb") as f:
            data = pickle.load(f)
        with open("docs/source/notebooks/mol_metadata_13CN.pkl", "rb") as f:
            metadata = pickle.load(f)

    data["GLO"] = 2 * data["F1l"] + 1

    _ = utils.supplement_molecule_data(data, metadata)
