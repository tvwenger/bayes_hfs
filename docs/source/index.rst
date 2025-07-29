bayes_hfs
=========

``bayes_hfs`` implements two models for molecular hyperfine spectroscopy. The first is ``HFSModel``, which is a general purpose model.
The second is ``HFSRatioModel``, which predicts observations of two species in order to infer the column density ratio. ``bayes_hfs`` 
is written in the ``bayes_spec`` Bayesian modeling framework, which provides methods to fit these models to data using Monte Carlo Markov Chain techniques.

Useful information can be found in the `bayes_hfs Github repository <https://github.com/tvwenger/bayes_hfs>`_, 
the `bayes_spec Github repository <https://github.com/tvwenger/bayes_spec>`_, and in the tutorials below.

============
Installation
============
.. code-block::

    conda create --name bayes_hfs -c conda-forge pymc pip
    conda activate bayes_hfs
    # Due to a bug in arviz, this fork is temporarily necessary
    # See: https://github.com/arviz-devs/arviz/issues/2437
    pip install git+https://github.com/tvwenger/arviz.git@plot_pair_reference_labels
    pip install bayes_hfs

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   notebooks/hfs_model
   notebooks/hfs_model_anomalies
   notebooks/hfs_ratio_model
   notebooks/cn_ratio_anomalies

.. toctree::
   :maxdepth: 2
   :caption: API:

   modules
