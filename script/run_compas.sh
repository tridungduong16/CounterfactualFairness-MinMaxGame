export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cudnn8.0-11.0/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cudnn8.0-11.0/lib64:$LIBRARY_PATH

learning_rate=0.01
ep=40
random_state=0
path=/home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/compas
lambda_path=/home/trduong/Data/counterfactual_fairness_game_theoric/reports/results/compas/lambda
for lambda_weight in 0.001 0.01
  do
    evaluate_path="${lambda_path}/evaluate-epoch-${ep}_lambda-${lambda_weight}_lr-${learning_rate}.csv"
    result_path="${lambda_path}/result-epoch-${ep}_lambda-${lambda_weight}_lr-${learning_rate}.csv"
    echo $path
    python /data/trduong/counterfactual_fairness_game_theoric/src/compas_train.py --epoch $ep --lambda_weight $lambda_weight --learning_rate $learning_rate
    python /data/trduong/counterfactual_fairness_game_theoric/src/compas_test.py
    python /data/trduong/counterfactual_fairness_game_theoric/src/baselines_classification.py --data_name compas
  done

