Counterfactual fairness under unknown causalknowledge: a game-theoretic approach
======================================================================

*Our work pay attention to counterfactual fairness with 

Author: `Dung Duong <https://scholar.google.com/citations?user=hoq2nt8AAAAJ&hl=en>`_, `Qian Li <https://scholar.google.com/citations?hl=en&user=yic0QMYAAAAJ>`_, `Guandong Xu <https://scholar.google.com/citations?user=kcrdCq4AAAAJ&hl=en&oi=ao>`_

This is the code used for the paper `Counterfactual fairness under unknown causalknowledge: a game-theoretic approach <https://arxiv.org/abs/2105.00703>`_.


How to run
-------------------------

Build the classifier model

.. code-block:: console

	python /multiobj-scm-cf/src/model_adult.py
	python /multiobj-scm-cf/src/model_credit.py
	python /multiobj-scm-cf/src/model_simple.py
	python /multiobj-scm-cf/src/model_sangiovese.py


Build the auto-encoder model

.. code-block:: console

	python /multiobj-scm-cf/src/dfencoder_adult.py
	python /multiobj-scm-cf/src/dfencoder_credit.py

Reproduce the results

.. code-block:: console

	python /multiobj-scm-cf/src/run_simplebn.py
	python /multiobj-scm-cf/src/run_adult.py
	python /multiobj-scm-cf/src/run_credit.py
	python /multiobj-scm-cf/src/run_sangiovese.py



