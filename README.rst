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


Citing
-------
If you find my work useful for your research work, please cite it as follows.

Duong, T. D., Li, Q., & Xu, G. (2022) Achieving Counterfactual Fairness with Imperfect Structural Causal Model
(Under review for Knowledge-based System)


Reference:
-------------------------

- Mahajan, D., Tan, C., & Sharma, A. (2019). Preserving causal constraints in counterfactual explanations for machine learning classifiers. arXiv preprint arXiv:1912.03277.
- Van Looveren, A., & Klaise, J. (2019). Interpretable counterfactual explanations guided by prototypes. arXiv preprint arXiv:1907.02584.
- AutoEncoders for DataFrames: https://github.com/AlliedToasters/dfencoder
