export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cudnn8.0-11.0/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LIBRARY_PATH

learning_rate=0.1
ep=423
random_state=0
path=/home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/law
lambda_path=/home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/law/lambda
for lambda_weight in 0.001 0.01 0.1 1 10 20 30 40 50 60 70 80 90 100
  do
    evaluate_path="${lambda_path}/evaluate-epoch-${ep}_lambda-${lambda_weight}_lr-${learning_rate}.csv"
    result_path="${lambda_path}/result-epoch-${ep}_lambda-${lambda_weight}_lr-${learning_rate}.csv"
    echo $path
    python /data/trduong/counterfactual_fairness_game_theoric/src/law_train.py --epoch $ep --lambda_weight $lambda_weight
    python /data/trduong/counterfactual_fairness_game_theoric/src/law_test.py
    python /data/trduong/counterfactual_fairness_game_theoric/src/law_evaluate.py
    cp /home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/evaluate_law.csv $evaluate_path
    cp /home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/law_ivr.csv $result_path
    echo $evaluate_path
    echo $result_path
  done

