#!/bin/bash

cd ..
effs=("experiment_configs/efficientnetb0_imagenet.json" "experiment_configs/efficientnetb1_imagenet.json" "experiment_configs/efficientnetb2_imagenet.json" "experiment_configs/efficientnetb3_imagenet.json" "experiment_configs/efficientnetb4_imagenet.json")
mobs=("experiment_configs/mobilenetv2-1.0-160_imagenet.json" "experiment_configs/mobilenetv2-1.0-192_imagenet.json" "experiment_configs/mobilenetv2-1.0-224_imagenet.json" "experiment_configs/mobilenetv2-1.3_224-imagenet.json" "experiment_configs/mobilenetv2-1.4-224_imagenet.json")


# train all models
for num in 1 2
do


    for conf in ${effs[@]}
    do
    python train_lightning.py $conf --seed $num --amp 1 --slurm 0
    done


    for conf in ${mobs[@]}
    do
    python train_lightning.py $conf --seed $num --amp 1 --slurm 0
    done

done
