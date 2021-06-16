export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cudnn8.0-11.0/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LIBRARY_PATH

ep=150
#lambda_weight=10
random_state=0
path=/home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/law
for lambda_weight in 0.001 0.01 0.1 1 10 20 30 40 50 60 70 80 90 100
  do
    evaluate_path="${path}/evaluate-epoch-${ep}_lambda-${lambda_weight}_random-${random_state}.csv"
    result_path="${path}/result-epoch-${ep}_lambda-${lambda_weight}_random-${random_state}.csv"
    echo $path
    python /data/trduong/counterfactual_fairness_game_theoric/src/law_train.py --epoch $ep --lambda_weight $lambda_weight
    python /data/trduong/counterfactual_fairness_game_theoric/src/law_test.py
    python /data/trduong/counterfactual_fairness_game_theoric/src/law_evaluate.py
    cp /home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/evaluate_law.csv $evaluate_path
    cp /home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/law_ivr.csv $result_path
  done

