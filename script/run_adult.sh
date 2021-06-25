export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cudnn8.0-11.0/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LIBRARY_PATH

learning_rate=0.001
ep=112
random_state=0
path=/home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/adult
lambda_path=/home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/adult/lambda
for random_state in 0 1 2 3 4 5 6 7 8 9 10
  do
    for lambda_weight in 10 20 30 40 50 60 70 80 90 100 1000 2000 3000
      do
        evaluate_path="${lambda_path}/evaluate/epoch-${ep}_lambda-${lambda_weight}_lr-${learning_rate}_random-${random_state}.csv"
        result_path="${lambda_path}/result/epoch-${ep}_lambda-${lambda_weight}_lr-${learning_rate}_random-${random_state}.csv"
        python /data/trduong/counterfactual_fairness_game_theoric/src/adult_train.py --epoch $ep --lambda_weight $lambda_weight --learning_rate $learning_rate
        python /data/trduong/counterfactual_fairness_game_theoric/src/adult_test.py
        python /data/trduong/counterfactual_fairness_game_theoric/src/evaluate_classifier.py --data_name adult
        cp /home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/evaluate_adult.csv $evaluate_path
        cp /home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/adult_ivr.csv $result_path
        rm -rf /home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/adult_ivr.csv
        rm -rf /home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/evaluate_adult.csv
        echo $evaluate_path
        echo $result_path
        clear
      done
  done