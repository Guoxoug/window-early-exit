"""
Get the latency and throughput of a single model, cascade.
Run after main evaluation as uncertainty performance is pulled from csv.
Dataloading time is normalised out for better consistency between systems. 
"""

import torch


import os
import json 
import pandas as pd
import numpy as np
import ast
import re

from models.model_utils import (
    model_generator, 
    load_weights_from_file
)

from time import time 


from utils.data_utils import (
    Data,
    get_preprocessing_transforms,
)


from tqdm import tqdm

from argparse import ArgumentParser

from utils.train_utils import get_filename

# argument parsing
parser = ArgumentParser()

parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)

parser.add_argument(
    "--seeds",
    default="12",
    type=str,
    help="random seed, can be specified as an arg or in the config."
)

parser.add_argument(
    "--batchsize",
    type=int,
    default=256,
    help="batchsize to use for offline throughput evaluation"
)

parser.add_argument(
    "--gpu",
    type=int,
    default=None,
    help="gpu override for debugging to set the gpu to use."
)

parser.add_argument(
    "--suffix",
    type=str,
    default="",
    help="added to end of filenames to differentiate them if needs be"
)


parser.add_argument(
    "--unc_task",
    type=str,
    default="cov@5",
    choices=[
        "cov@5", 
        "cov@10",
        "risk@50",
        "risk@80",
        "FPR@80",
        "FPR@95"
    ]
)

parser.add_argument(
    "--exit_metric",
    type=str,
    default="confidence"
)


args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)

if args.seeds is not None:
    seeds = list(args.seeds)
elif args.secondary_config_path is not None:
    seeds = [0,0]
else:
    seeds = [0,1] 


# set gpu
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        config["gpu_id"]
    ).replace("[", "").replace("]", "")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {dev}")

LAT_EARLY_STOP = 2000 # set to -1 to eval whole test set, quite slow

results_path = os.path.join(
    config["test_params"]["results_savedir"], 
    get_filename(config, seed=None)
)

csv_path = os.path.join(
    results_path, 
    f"{config['model']['model_type']}_{config['id_dataset']['name']}_cascade_{seeds[0]}_{seeds[1]}_window_threshold_strategy_{args.exit_metric}_{args.unc_task}_.csv"
)
df = pd.read_csv(csv_path)


def get_df_col(id_data_name, unc_task, unc_name, ood_data_name=None):
    unc_name = unc_name+"_ens"

    if "cov" in unc_task:
        return f"{id_data_name} {unc_task} {unc_name}"
    elif "risk" in unc_task:
        if ood_data_name is None:
            return f"{id_data_name} {unc_task} {unc_name}"
        else:
            return f"{id_data_name} + {ood_data_name} {unc_task} {unc_name}"
    elif "FPR" in unc_task:
        return f"OOD {ood_data_name} {unc_task} {unc_name}"

unc_col = get_df_col(config["id_dataset"]["name"], args.unc_task, args.exit_metric)
df = df[[
    unc_col, 
    "window"
]]


# this is actually a confidence as we are using MSP
# MSP is aliased to confidence
unc_func = lambda logits: logits.softmax(dim=-1).max(dim=-1).values

models = []
for seed in seeds:
    # where pretrained weights are

    # load base model
    model = model_generator(
        config["model"]["model_type"],
        **config["model"]["model_params"]
    )
    weights_path = os.path.join(
        config["model"]["weights_path"],
        get_filename(config, seed=seed) + ".pth"
    )

    print(f"Trying to load weights from: {weights_path}\n")
    load_weights_from_file(model, weights_path)
    print("Loading successful")
    model.eval()
    model.to(dev)
    models.append(model)

transforms = get_preprocessing_transforms(
    config["id_dataset"]["name"], 
    resolution=model.resolution # different resolutions effect computation
)

test_data = Data(
    config["id_dataset"]["name"],
    config["id_dataset"]["datapath"],
    batch_size=args.batchsize,
    test_batch_size=args.batchsize,
    num_workers=16,
    idx_path=config["id_dataset"]["idx_path"],
    transforms=transforms
)


# random sampling 
test_throughput_loader = torch.utils.data.DataLoader(
    test_data.test_set, 
    batch_size=args.batchsize, shuffle=True, num_workers=16
)

test_latency_loader = torch.utils.data.DataLoader(
    test_data.test_set, 
    batch_size=1, shuffle=True, num_workers=16
)


# Get computational results ====================================================

result_rows = []

# singe model ------------------------------------------------------------------
results = {"model": "single model"}
results[args.unc_task] = ast.literal_eval(df[unc_col].iloc[0])[0]
print(results)
single_model = models[0]
throughputs = []
with torch.no_grad():
    for i in range(5):
        not_exited_list = []
        dataset_time = 0.0
        for inputs, labels in tqdm(test_throughput_loader):
            batch_time = time()
            inputs, labels = inputs.to(dev), labels.to(dev)
            out = models[0](inputs)
            batch_end_time = time()
            batched_time_taken = batch_end_time - batch_time
            dataset_time += batched_time_taken

        throughput = len(test_data.test_set)/dataset_time
        throughputs.append(throughput)

throughput = np.array(throughputs).mean()
print(f"single model throughput: {throughput:.1f} images/second")
results["throughput"] = throughput

models = [model.to("cpu") for model in models]
print("evaluating latency on CPU")
lats = []
with torch.no_grad():
    for i in range(1):
        dataset_time = 0.0
        # normalise for data loading time
        for j, (inputs, labels) in tqdm(enumerate(test_latency_loader)):
            # this takes a very long time, can terminate early for convenience
            if j == LAT_EARLY_STOP:
                break
            single_time = time()
            out = single_model(inputs)
            single_end_time = time()
            single_time_taken = single_end_time - single_time
            dataset_time += single_time_taken
        if j < 0 or j> len(test_data.test_set):
            av_latency = dataset_time/len(test_data.test_set)
        else:
            av_latency = dataset_time/j
        lats.append(av_latency)
latency = np.array(lats).mean()

print(f"single model average latency: {latency:.4g} seconds")
results["latency"] = latency
print(results)
result_rows.append(results)

# cascade ----------------------------------------------------------------------
results = {"model": "cascade"}
results[args.unc_task] = ast.literal_eval(df[unc_col].iloc[1])[0]

try:
    window = ast.literal_eval(
        re.search(
            "\(([^)]+), dtype",
            df["window"].iloc[1]
        ).group(1)
    )
    print(window)
except:
    window = ast.literal_eval(df["window"].iloc[1])
print(results)


print("evaluating batched throughput on GPU")
# iterate over all data offline, filter, then iterate over remaining data
models = [model.to(dev) for model in models]
throughputs = []
with torch.no_grad():
    for i in range(5):
        not_exited_list = []
        dataset_time = 0.0
        for inputs, labels in tqdm(test_throughput_loader):
            # normalise out dataloading time
            batch_time = time()
            inputs, labels = inputs.to(dev), labels.to(dev)
            out = models[0](inputs)
            uncs = unc_func(out)
            exited = (uncs > window[1]) + (uncs < window[0])
            not_exited = ~exited
            out = models[1](inputs[not_exited]) # 2nd model on subset
            batch_end_time = time()
            batched_time_taken = batch_end_time - batch_time
            dataset_time += batched_time_taken

        throughput = len(test_data.test_set)/dataset_time
        throughputs.append(throughput)

throughput = np.array(throughputs).mean()
print(f"cascaded model throughput: {throughput:.1f} images/second")
results["throughput"] = throughput


print("evaluating latency on CPU")
models = [model.to("cpu") for model in models]
lats = []
with torch.no_grad():
    for i in range(1):
        dataset_time = 0.0
        for j, (inputs, labels) in tqdm(enumerate(test_latency_loader)):
            if j == LAT_EARLY_STOP:
                break
            single_time = time()
            out = models[0](inputs)
            uncs = unc_func(out)
            if not ((uncs > window[1]) + (uncs < window[0])).item():
                out = models[1](inputs)
            single_end_time = time()
            single_time_taken = single_end_time - single_time
            dataset_time += single_time_taken
        if j < 0 or j> len(test_data.test_set):
            av_latency = dataset_time/len(test_data.test_set)
        else:
            av_latency = dataset_time/j
        lats.append(av_latency)
latency = np.array(lats).mean()

print(f"cascaded model average latency: {latency:.4g} seconds")
results["latency"] = latency
print(results)
result_rows.append(results)



# save results as csv

results_df = pd.DataFrame(
    result_rows
)


spec = get_filename(config, seed=None)
filename = get_filename(config) + f"{seeds[0]}_{seeds[1]}_latency_throughput_{args.unc_task}_"
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
savepath = os.path.join(save_dir, f"{filename}{args.suffix}.csv")

# just overwrite what's there
results_df.to_csv(savepath, mode="w", header=True)
print(results_df)
