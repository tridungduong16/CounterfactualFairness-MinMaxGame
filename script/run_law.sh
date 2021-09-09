export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cudnn8.0-11.0/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LIBRARY_PATH

learning_rate=0.01
ep=400
random_state=0
path=/Data/counterfactual_fairness_game_theoric/reports/results/law
lambda_path=/Data/counterfactual_fairness_game_theoric/reports/results/law/lambda
for random_state in 0 1 2 3 4 5 6 7 8 9 10
  do
    for lambda_weight in 0.001 0.01 0.1 1 10 20 30 40 50 60 70 80 90 100
      do
        evaluate_path="${lambda_path}/evaluate/epoch-${ep}_lambda-${lambda_weight}_lr-${learning_rate}_random-${random_state}.csv"
        result_path="${lambda_path}/result/epoch-${ep}_lambda-${lambda_weight}_lr-${learning_rate}_random-${random_state}.csv"
        echo $path
        python /counterfactual_fairness_game_theoric/src/law_train.py --epoch $ep --lambda_weight $lambda_weight
        python /counterfactual_fairness_game_theoric/src/law_test.py
        python /counterfactual_fairness_game_theoric/src/law_evaluate.py
        cp /Data/counterfactual_fairness_game_theoric/reports/results/evaluate_law.csv $evaluate_path
        cp /Data/counterfactual_fairness_game_theoric/reports/results/law_ivr.csv $result_path
        rm -rf /Data/counterfactual_fairness_game_theoric/reports/results/law_ivr.csv
        rm -rf /Data/counterfactual_fairness_game_theoric/reports/results/evaluate_law.csv
        echo $evaluate_path
        echo $result_path
        clear
      done
  done

