export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cudnn8.0-11.0/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LIBRARY_PATH

learning_rate=0.01
ep=400
random_state=0
path=/home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/compas
lambda_path=/home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/compas/lambda
for lambda_weight in 0.001 0.01 0.1 1 10 20 30 40 50 60 70 80 90 100
  do
    evaluate_path="${lambda_path}/evaluate/epoch-${ep}_lambda-${lambda_weight}_lr-${learning_rate}.csv"
    result_path="${lambda_path}/result/epoch-${ep}_lambda-${lambda_weight}_lr-${learning_rate}.csv"
    python /data/trduong/counterfactual_fairness_game_theoric/src/compas_train.py --epoch $ep --lambda_weight $lambda_weight --learning_rate $learning_rate
    python /data/trduong/counterfactual_fairness_game_theoric/src/compas_test.py
    python /data/trduong/counterfactual_fairness_game_theoric/src/evaluate_classifier.py --data_name compas
    cp /home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/evaluate_compas.csv $evaluate_path
    cp /home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/compas_ivr.csv $result_path
    rm -rf /home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/compas_ivr.csv
    rm -rf /home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/evaluate_compas.csv
    echo $evaluate_path
    echo $result_path
  done

