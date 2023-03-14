#!/bin/bash
cd ..

# note run with 2>&1 | tee output.txt to get output for debugging


gpu_num=0

effs=("experiment_configs/efficientnetb0_imagenet.json" "experiment_configs/efficientnetb1_imagenet.json" "experiment_configs/efficientnetb2_imagenet.json" "experiment_configs/efficientnetb3_imagenet.json" "experiment_configs/efficientnetb4_imagenet.json")
for num in 12 21
do


    for conf in ${effs[@]}
    do
    python eval_latency_throughput.py $conf --seeds $num --gpu $gpu_num --batchsize 256 --unc_task cov@5
    done


done

python plot_unc_lat_throughput.py \
    experiment_configs/efficientnetb0_imagenet.json \
    --model_family efficientnet \
    --seeds 12 \
    --unc_task cov@5 \
    --exit_metric confidence \
    --strategy_func window_threshold_strategy

echo "finished testing"