#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

for i in 0.1 0.5 1 1.5 2 2.5 3 3.5 4 4.5
  do
    echo "Run for law"
    echo "Lambda $i"
    python /home/trduong/Data/counterfactual_fairness_game_theoric/src/law_train.py --lambda_weight $i --run_lambda
    python /data/trduong/counterfactual_fairness_game_theoric/src/law_test.py --lambda_weight $i --run_lambda
  done

echo "-------------------- Running script done ------------------------------"