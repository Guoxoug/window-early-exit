
from utils.data_utils import DATA_NAME_MAPPING
import pandas as pd

pd.set_option("display.max_rows", 200,
              "display.max_columns", 10)
import numpy as np
from argparse import ArgumentParser
import json
from utils.train_utils import get_filename
from utils.eval_utils import METRIC_NAME_MAPPING
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import ast
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
    default=None,
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
    results_path = config["test_params"]["results_savedir"], 



if args.model_family == "efficientnet":
    model_types = [f"efficientnetb{i}" for i in range(5)]
elif args.model_family == "mobilenetv2":
    model_types = [f"mobilenetv2-{spec}" for spec in ["1.0-160", "1.0-192","1.0-224","1.3-224","1.4-224",]]

def get_df_cols(id_data_name, unc_task, unc_name, ood_data_name=None):
    if args.ensemble and "confidence" in unc_name:
        unc_name = unc_name+"_ens"
    cols=[]

    # first
    if "cov" in unc_task:
        cols.append(f"{id_data_name} {unc_task} {unc_name}")
    elif "risk" in unc_task:
        if ood_data_name is None:
            cols.append(f"{id_data_name} {unc_task} {unc_name}")
        else:

            cols.append(f"{id_data_name} + {ood_data_name} {unc_task} {unc_name}")
            cols.append(f"{id_data_name} + {ood_data_name} {unc_task} {unc_name} adj")
    elif "FPR" in unc_task:
        assert ood_data_name is not None
        cols.append(f"OOD {ood_data_name} {unc_task} {unc_name}")
        cols.append(f"OOD {ood_data_name} {unc_task} {unc_name} adj")


    if ood_data_name is not None:
        

        cols.append(f"{ood_data_name}_average_macs_adj")
        cols.append(f"{ood_data_name}_average_macs") # -3
        
        cols.append(f"{id_data_name}_average_macs_adj")

    # last
    cols.append(f"{id_data_name}_average_macs")
    return cols 


def clean_list(x):
    """Clean dataframe columns of lists.
    csv will be read as strings, we want the first element of each list [value, threshold]
    """

    if type(x) == str:
        return ast.literal_eval(x)[0]
    else:
        return x



cols_to_extract = get_df_cols(
    configs[0]["id_dataset"]["name"],
    args.unc_task,
    args.exit_metric,
    ood_data_name=args.ood_data
)

dfs = []
print(results_path)

for i,model in enumerate(model_types):
    dfss = []
    for seed_order in [seeds, seeds[::-1]]:
        # will try to average over both directions is results are there
        try:
            df0 = pd.read_csv(
                os.path.join(
                    config["test_params"]["results_savedir"],
                    f"{model}_{configs[0]['id_dataset']['name']}/{model}_{configs[0]['id_dataset']['name']}_cascade_{seed_order[0]}_{seed_order[1]}_{args.strategy_func}_{args.exit_metric}_{args.unc_task}_.csv"
                ),
                index_col=0
            )[cols_to_extract].applymap(clean_list)
            dfss.append(df0)
            print(f"seed order {seed_order} loaded")
        except:
            print(f"failed to load seed order {seed_order}")
            # so you can plot if only run in one direction
            pass
        
    dfs.append(dfss)




neww_dfs = [pd.concat(df_list) for df_list in dfs]
# take average over directions
new_dfs = [df.groupby(df.index).mean() for df in neww_dfs]


id_macs = cols_to_extract[-1]
single_macs = [df[id_macs].iloc[0] for df in new_dfs] # 1st
ens_macs = [df[id_macs].iloc[-1] for df in new_dfs] # last
cascade_macs = [df[id_macs].iloc[1] for df in new_dfs] 


perf = cols_to_extract[0]
single_perf = [df[perf].iloc[0] for df in new_dfs] # 1st
ens_perf = [df[perf].iloc[-1] for df in new_dfs] # last

cascade_perf = [df[perf].iloc[1] for df in new_dfs] 

if args.ood_data is not None:
    ood_macs = cols_to_extract[-3]
    single_ood_macs = [df[ood_macs].iloc[0] for df in new_dfs] # 1st
    ens_ood_macs = [df[ood_macs].iloc[-1] for df in new_dfs] # last
    cascade_ood_macs = [df[ood_macs].iloc[1] for df in new_dfs] 

    # assume problem setting where the volume of OOD data is the same 
    # as ID
    cascade_macs = 0.5 * (np.array(cascade_macs) + np.array(cascade_ood_macs))


markers = ["o", "x", "+"]
fig, ax = plt.subplots(1,1,figsize=(6,3), sharex=True)
# ax.set_xscale("log")
ax.plot(single_macs, single_perf, marker=markers[0], color="black", label="single model")
ax.plot(ens_macs, ens_perf, marker=markers[1], color="indianred", label="ensemble")
ax.plot(cascade_macs, cascade_perf, marker=markers[2], color="green", label="window-based cascade (ours)")
if args.ood_data is not None:
    cascade_macs = [df[f"{id_macs}_adj"].iloc[1] for df in new_dfs] 
    cascade_ood_macs = [df[f"{ood_macs}_adj"].iloc[1] for df in new_dfs] 

    # average as we assume 1:1 ratio
    # adjusted increase should be 20% in MACs for the cascade 
    cascade_macs = (np.array(cascade_macs)+np.array(cascade_ood_macs))/2
    cascade_perf = [df[f"{perf} adj"].iloc[1] for df in new_dfs] 
    ax.plot(cascade_macs, cascade_perf, marker=markers[2], color="green", label="adjusted window (ours)", linestyle=":")
ax.annotate(
    MODEL_NAME_MAPPING[model_types[0]], 
    xy = (single_macs[0], single_perf[0]),
    textcoords="offset pixels",
    xytext=(10, 0),
)
ax.annotate(
    MODEL_NAME_MAPPING[model_types[-1]], 
    xy = (single_macs[-1], single_perf[-1]),
    textcoords="offset pixels",
    xytext=(10, 0),
)
ax.set_xscale("log")
ax.legend()
ax.set_ylabel(EVAL_MAPPING[args.unc_task])
ax.set_xlabel("Average MACs")
ax.grid(visible=True, which='minor', color='w', linewidth=0.5, axis="x")
fig.tight_layout()


 # specify filename
ood = "" if args.ood_data is None else "_" + args.ood_data
spec = get_filename(config, seed=None)
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
filename = get_filename(config, seed=config["seed"]) +  \
    f"{ood}_cascade_{args.strategy_func}_{args.unc_task}_macs_{args.exit_metric}.pdf"
path = os.path.join(save_dir, filename)
fig.savefig(path)
print(f"figure saved to:\n{path}")



