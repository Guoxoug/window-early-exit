#!/bin/bash


cd ..

# path the configuration file
config_path=experiment_configs/msdnet_imagenet.json
weights=models/saved_models/msdnet_pretrained/step=7/msdnet-step=7-block=5.pth.tar
gpu=0

# weights path MUST be specified
python test_msdnet.py $config_path --seed 0 --gpu $gpu --weights_path $weights
python plot_msdnet_unc_cascade.py $config_path --seed 0 --weights_path $weights