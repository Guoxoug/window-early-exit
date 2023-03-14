#!/bin/bash

cd ..

conf=experiment_configs/efficientnetb0_imagenet.json
family=efficientnet

python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task cov@5
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task cov@10
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@50
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@80 
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@50 --ood_data openimage-o
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@80 --ood_data openimage-o
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@50 --ood_data inaturalist
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@80   --ood_data inaturalist
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric energy --unc_task FPR@80 --ood_data openimage-o
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric energy --unc_task FPR@95 --ood_data openimage-o
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric energy --unc_task FPR@80 --ood_data inaturalist
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric energy --unc_task FPR@95 --ood_data inaturalist

python plot_policy_comparison.py experiment_configs/efficientnetb2_imagenet.json  --seeds 12 


conf=experiment_configs/mobilenetv2-1.0-160_imagenet.json
family=mobilenetv2

python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task cov@5
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task cov@10
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@50
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@80 
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@50 --ood_data openimage-o
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@80 --ood_data openimage-o
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@50 --ood_data inaturalist
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric confidence --unc_task risk@80   --ood_data inaturalist
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric energy --unc_task FPR@80 --ood_data openimage-o
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric energy --unc_task FPR@95 --ood_data openimage-o
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric energy --unc_task FPR@80 --ood_data inaturalist
python plot_unc_macs_comparison.py $conf  --model_family $family --seeds 12  --strategy_func window_threshold_strategy --exit_metric energy --unc_task FPR@95 --ood_data inaturalist