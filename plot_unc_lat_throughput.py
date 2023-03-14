import argparse
from utils.data_utils import DATA_NAME_MAPPING
import pandas as pd

pd.set_option("display.max_rows", 200,
              "display.max_columns", 10)
import numpy as np
from argparse import ArgumentParser
import json
from utils.train_utils import get_filename
import os
import matplotlib.pyplot as plt
import seaborn as sns
from models.model_utils import MODEL_NAME_MAPPING
sns.set_theme()
parser = ArgumentParser()
parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)

parser.add_argument(
    "--model_family",
    default="efficientnet",
    choices=["efficientnet", "mobilenetv2"]
)

parser.add_argument(
    "--seeds",
    default="12",
    type=str,
    help="string containing random seeds of the two cascade members in order."
)
parser.add_argument(
    "--results_path",
    type=str,
    default=None,
    help=(
        "directory where result .csv files are kept," 
        "deduced from config by default"
    )
)

parser.add_argument(
    "--secondary_config_path",
    default=None,
    type=str,
    help="if cascade is heterogeneous then this points to the second model"
)

parser.add_argument(
    "--suffix",
    default="",
    help="a suffix to differentiate a file"
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
parser.add_argument(
    "--strategy_func",
    type=str,
    default="window_threshold_strategy"
)

parser.add_argument(
    "--ensemble",
    default=1,
    help="whether to ensemble the models together on the second stage"
)

parser.add_argument(
    "--ood_data",
    type=str,
    default=None,
    help="ood dataset name"
)


args = parser.parse_args()

EVAL_MAPPING = {
    "cov@5": r"Cov@5$\rightarrow$",
    "cov@10": r"Cov@10$\rightarrow$",
    "risk@50": r"Risk@50$\leftarrow$",
    "risk@80": r"Risk@80$\leftarrow$",
    "FPR@95": r"FPR@95$\leftarrow$",
    "FPR@80": r"FPR@80$\leftarrow$"
}

args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)

if args.secondary_config_path is not None:
    secondary_config = open(args.secondary_config_path)
    secondary_config = json.load(secondary_config)

# default is to assume the cascade is heterogeneous 
else:
    secondary_config = config

configs = [config, secondary_config]

if args.seeds is not None:
    seeds = list(args.seeds)
elif args.secondary_config_path is not None:
    seeds = [1,1]
else:
    seeds = [1,2] 

# results path generated as results_savedir/arch_dataset
if args.results_path is not None:
    results_path = args.results_path
else:
    results_path = config["test_params"]["results_savedir"]


if args.model_family == "efficientnet":
    model_types = [f"efficientnetb{i}" for i in range(5)]
elif args.model_family == "mobilenetv2":
    model_types = [f"mobilenetv2_{spec}" for spec in ["1.0_160", "1.0_192","1.0_224","1.3_224","1.4_224",]]


dfs = []
for i,model in enumerate(model_types):
    dfss = []
    for seed_order in [seeds, seeds[::-1]]:
        # will try to average over both directions is results are there
        try:
            df0 = pd.read_csv(
                os.path.join(
                    results_path,
                    f"{model}_{configs[0]['id_dataset']['name']}/{model}_{configs[0]['id_dataset']['name']}{seed_order[0]}_{seed_order[1]}_latency_throughput_{args.unc_task}_.csv",
                
                ), index_col=0
            )
            dfss.append(df0)
            print(f"seed order {seed_order} loaded")
        except:
            print(f"failed to load seed order {seed_order}")
            pass
        
    dfs.append(dfss)




new_dfs = [pd.concat(df_list) for df_list in dfs]
# take average
new_dfs = [df.groupby(df.index).mean() for df in new_dfs]


single_throughput = [df["throughput"].iloc[0] for df in new_dfs] # 1st
cascade_throughput = [df["throughput"].iloc[1] for df in new_dfs] 

single_lat = [df["latency"].iloc[0] for df in new_dfs] # 1st
cascade_lat = [df["latency"].iloc[1] for df in new_dfs] 

perf = args.unc_task
single_perf = [df[perf].iloc[0] for df in new_dfs] # 1st
cascade_perf = [df[perf].iloc[1] for df in new_dfs] 


markers = ["o", "x", "+"]
fig, ax = plt.subplots(1,2,figsize=(6,2.5), sharey=True)

ax[0].plot(single_throughput, single_perf, marker=markers[0], color="black", label="single model")
ax[0].plot(cascade_throughput, cascade_perf, marker=markers[2], color="green", label="window-based\ncascade (ours)")


ax[0].set_ylabel(EVAL_MAPPING[args.unc_task])
ax[0].set_xlabel("thoughput (img/s)")
ax[0].grid(visible=True, which='minor', color='w', linewidth=0.5, axis="x")
ax[1].plot(np.array(single_lat)*1000, single_perf, marker=markers[0], color="black", label="single model")
ax[1].plot(np.array(cascade_lat)*1000, cascade_perf, marker=markers[2], color="green", label="window-based\ncascade (ours)")
ax[1].set_xlabel("latency (ms)")
ax[0].set_xlim(xmin=0)

ax[1].grid(visible=True, which='minor', color='w', linewidth=0.5, axis="x")
ax[1].legend()
fig.tight_layout()


 # specify filename
spec = get_filename(config, seed=None)
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
filename = get_filename(config, seed=config["seed"]) +  \
    f"_cascade_{args.unc_task}_latency_throughput_{args.exit_metric}.pdf"
path = os.path.join(save_dir, filename)
fig.savefig(path)
print(f"figure saved to:\n{path}")



