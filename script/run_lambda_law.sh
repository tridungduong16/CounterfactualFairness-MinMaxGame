#!/bin/bash
# 0.1 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10 20 30 40 50
echo "Bash version ${BASH_VERSION}..."

var_lambda="0.1 0.5 1 1.5 2 2.5 3 3.5 4"

#for i in 0.1 0.5 1 1.5 2 2.5 3 3.5 4
#  do
#    echo "Run for law"
#    echo "Lambda $i"
#    python /home/trduong/Data/counterfactual_fairness_game_theoric/src/law_train.py --lambda_weight $i --run_lambda
#  done
echo "Fucking up"
echo $var_lambda

python /data/trduong/counterfactual_fairness_game_theoric/src/law_test.py --lambda_weight "$var_lambda" --run_lambda
python /data/trduong/counterfactual_fairness_game_theoric/src/law_evaluate.py --lambda_weight "$var_lambda" --run_lambda

echo "-------------------- Running script done ------------------------------"