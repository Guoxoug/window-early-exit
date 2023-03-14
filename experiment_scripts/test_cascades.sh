#!/bin/bash
cd ..

# note run with 2>&1 | tee output.txt to get output for debugging


gpu_num=0

effs=("experiment_configs/efficientnetb0_imagenet.json" "experiment_configs/efficientnetb1_imagenet.json" "experiment_configs/efficientnetb2_imagenet.json" "experiment_configs/efficientnetb3_imagenet.json" "experiment_configs/efficientnetb4_imagenet.json")
mobs=("experiment_configs/mobilenetv2-1.0-160_imagenet.json" "experiment_configs/mobilenetv2-1.0-192_imagenet.json" "experiment_configs/mobilenetv2-1.0-224_imagenet.json" "experiment_configs/mobilenetv2-1.3-224_imagenet.json" "experiment_configs/mobilenetv2-1.4-224_imagenet.json")

for num in 1 2
do


    for conf in ${mobs[@]}
    do
    python test.py $conf --seed $num --gpu $gpu_num
    done
    
    for conf in ${effs[@]}
    do
    python test.py $conf --seed $num --gpu $gpu_num
    done



done






# windows, many windows
for num in 12 21 # both directions
do

    for conf in ${mobs[@]}
    do
    python cascade_results.py $conf  --seeds $num  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task cov@5
    python cascade_results.py $conf  --seeds $num  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task cov@10
    python cascade_results.py $conf  --seeds $num  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@50
    python cascade_results.py $conf  --seeds $num  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@80
    python cascade_results.py $conf  --seeds $num  --strategy_func window_threshold_strategy --exit_metric energy --unc_task FPR@80
    python cascade_results.py $conf  --seeds $num  --strategy_func window_threshold_strategy --exit_metric energy --unc_task FPR@95
    done

    for conf in ${effs[@]}
    do
    python cascade_results.py $conf  --seeds $num  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task cov@5
    python cascade_results.py $conf  --seeds $num  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task cov@10
    python cascade_results.py $conf  --seeds $num  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@50
    python cascade_results.py $conf  --seeds $num  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@80
    python cascade_results.py $conf  --seeds $num  --strategy_func window_threshold_strategy --exit_metric energy --unc_task FPR@80
    python cascade_results.py $conf  --seeds $num  --strategy_func window_threshold_strategy --exit_metric energy --unc_task FPR@95
    done

done



# single threshold
for num in 12 21 # both directions
do

    for conf in ${effs[@]}
    do
    python cascade_results.py $conf --seeds $num   --strategy_func single_threshold_strategy --exit_metric confidence
    done

    for conf in ${mobs[@]}
    do
    python cascade_results.py $conf --seeds $num   --strategy_func single_threshold_strategy --exit_metric confidence
    done

done



echo "finished testing"