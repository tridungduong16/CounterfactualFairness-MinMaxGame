Counterfactual fairness under unknown causalknowledge: a game-theoretic approach
======================================================================

*Our work pay attention to counterfactual fairness with unknown structural causal model

How to run
-------------------------

Generate the latent features for counterfactual fairness baselines

.. code-block:: console
    python src/law_baselines.py --generate
    python src/baselines_classification.py --data_name adult --generate
    python src/baselines_classification.py --data_name compas --generate

Run the baselines method

.. code-block:: console

    python src/law_baselines.py
    python src/baselines_classification.py --data_name adult
    python src/baselines_classification.py --data_name compas

Run the our method

.. code-block:: console

    bash /script/run_adult.sh
    bash /script/run_compas.sh
    bash /script/run_law.sh
