
import pandas as pd

pd.set_option("display.max_rows", 200,
              "display.max_columns", 10)
import numpy as np
from argparse import ArgumentParser
import json
from utils.train_utils import get_filename
from utils.eval_utils import *
from utils.data_utils import *
from utils.cascade_utils import *
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore

sns.set_theme()
parser = ArgumentParser()
parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
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
    "--seed",
    type=int,
    default=0
)

parser.add_argument(
    "--exit_metric",
    type=str,
    default="confidence"
)
parser.add_argument(
    "--weights_path",
    type=str,
    default=None,
    help="Optional path to weights, overrides config."
)

args = parser.parse_args()

metrics = ["errFPR@95", " FPR@95"]

EVAL_MAPPING = {
    "cov@5": r"Cov@5$\rightarrow$",
    "cov@10": r"Cov@10$\rightarrow$",
    "risk@50": r"Risk@50$\leftarrow$",
    "risk@80": r"Risk@80$\leftarrow$",
    "FPR@95": r"FPR@95$\leftarrow$",
    "FPR@80": r"FPR@80$\leftarrow$"
}

# load config
config = open(args.config_path)
config = json.load(config)


# results path generated as results_savedir/arch_dataset
if args.results_path is not None:
    results_path = args.results_path
else:
    results_path = os.path.join(
        config["test_params"]["results_savedir"], 
        get_filename(config, seed=None)
    )


if args.weights_path is not None and config["model"]["model_type"] == "msdnet":
    idx_path = os.path.join(
        os.path.split(args.weights_path)[0],
        "index.pth"
    )
    indices = torch.load(idx_path)
    print("loaded MSDNet indices")
else:
    indices = None


id_data = Data(
    **config["id_dataset"],
    test_only=False,
    transforms=get_preprocessing_transforms(config["id_dataset"]["name"]),
    indices=indices
)

test_loader = id_data.test_loader

# results path generated as results_savedir/arch_dataset
results_path = os.path.join(
    config["test_params"]["results_savedir"],
    get_filename(config, seed=None)
)
logits_path = os.path.join(
        results_path, get_filename(config, seed=args.seed) + "_logits.pth"
    )  


print("Loading logits")
logits_dict = torch.load(logits_path)
print("Loading complete")

logits_dict = logits_dict["afp, wfp"]
print(logits_dict.keys())
# load information about flops/MACs
cost_path = os.path.join(
    results_path, get_filename(config) + "_comp.csv"
)  # results_savedir/arch_dataset/arch_dataset_seed_logits.pth
cost_df = pd.read_csv(cost_path, index_col=[0]).transpose()


macs_dict = dict(cost_df["MACs"])

print(macs_dict)

heads_of_interest = ["head2", "head3", "head5"]
logits = [logits_dict[id_data.name][head] for head in heads_of_interest]
macs_list = [macs_dict[head] for head in heads_of_interest]

def eval_id_measures(log_probs, labels):
    """Evaluate some standard measures of ID performance"""
    top1 = TopKError(k=1, percent=True)
    top5 = TopKError(k=5, percent=True)
    nll = torch.nn.CrossEntropyLoss()
    results = {}
    
    results["top1"] = top1(labels, log_probs)
    results["top5"] = top5(labels, log_probs)
    results["nll"] = nll(log_probs, labels).item()  # backwards

    return(results)


def evaluate_early_exit(
    logits, labelled_data, strategy_func,
    cascade_macs,
    data_name=None,
    strategy_kwargs=[], 
):
    """Evaluate the cascade's performance."""

    n_exits=len(logits)
    if data_name is None:
        data_name = labelled_data.name
    print(f"eval on: {data_name}")
    try:
        labels = torch.tensor(labelled_data.test_set.targets)
    except:
        labels = torch.tensor(id_data.test_set.targets)


    # results =================================================================

    results = {}
    results["dataset"] = data_name
    results["strategy"] = strategy_func.__name__

    # ID stuff ----------------------------------------------------------------
    # get uncertainties for each member
    # no ensembling for MSDNet as exits are already of different 
    # accuracies 
    id_metrics = [
        uncertainties(
            logits[i]
        )
        for i in range(n_exits)
    ]

    id_logits  = logits


    # start from the deepest
    # means we don't have to work on subsets of the data
    final_id_logits = id_logits[-1]
    # id evaluation
    for j in range(n_exits-1, 0, -1):
        final_id_logits = strategy_func(
            id_metrics[j-1],
            id_logits[j-1], final_id_logits,
            **strategy_kwargs[j-1]
        )

    macs = [
        torch.ones(len(final_id_logits))*cascade_macs[i]
        for i in range(len(cascade_macs))
    ]
    final_id_macs = macs[-1]
    for j in range(n_exits-1, 0, -1):
        final_id_macs = strategy_func(
            id_metrics[j-1],
            macs[j-1], final_id_macs,
            **strategy_kwargs[j-1]
        )
    results[f"{data_name}_average_macs"] = final_id_macs.mean().item()
    results.update(
        eval_id_measures(final_id_logits, labels)
    )

    final_id_metrics = uncertainties(final_id_logits)


    # AUROC for misclassification detection
    max_logits, preds = final_id_logits.max(dim=-1)
    miscls_labels = (preds != labels)

    correct_idx = (miscls_labels == 0)

    # selective classification metrics
    sc_eval_modes = ["AURC", "cov@5", "cov@10", "risk@50", "risk@80"]
    for eval_mode in sc_eval_modes:
        res = scod_results(
            final_id_metrics, correct_idx, mode=eval_mode
        )

        res = {
            f"{data_name} {eval_mode} " + k: v
            for k, v in res.items()
            if k != "mode"
        }
        results.update(res)


    return results



# results for going from head3 to head5
# plot how uncertainty performance changes as computation increases ============
val_logits =  [
    logits_dict[f"{id_data.name}_val"][head] 
    for head in heads_of_interest
]
val_labels = torch.tensor(id_data.val_set.targets)

val_correct_idx = [
    val_labels == val_logits[i].argmax(dim=-1)
    for i in range(len(val_logits))
]
print(f"accuracy: {val_correct_idx[1].to(float).mean().item()}")
id_uncs = [
    uncertainties(
        logits=val_logits[i]
    ) for i in range(len(val_logits))
]



id_confs = id_uncs[1]["confidence"] # MSP of head 3 for 3->5

# get windows
curve_reqs = {
    "cov@5":{"curve":"RC", "requirement": ["risk", 0.05]}, 
    "cov@10":{"curve":"RC", "requirement": ["risk", 0.10]}, 
    "risk@50":{"curve":"RC", "requirement": ["coverage", 0.5]}, 
    "risk@80": {"curve":"RC", "requirement":["coverage", 0.8]}, 
    "FPR@80":{"curve":"ROC", "requirement": ["tpr", 0.8]}, 
    "FPR@95": {"curve":"ROC", "requirement":["tpr", 0.95]}
}

curve_req = curve_reqs[args.unc_task]
if curve_req["curve"] == "RC":
    tau = get_threshold(
        val_correct_idx[1], id_confs, 
        curve="RC", requirement=curve_req["requirement"]
    )
elif curve_req["curve"] == "ROC":
    # generate random OOD data, TPR should be independent
    tau = get_threshold(
        torch.cat([torch.ones(len(id_confs)), torch.zeros(len(id_confs))]), 
        torch.cat([id_confs, id_confs + torch.randn(len(id_confs))]), 
        curve="ROC", requirement=curve_req["requirement"]
    )
tau_quantile = percentileofscore(id_confs, tau)/100.0
exit_ts = [
    torch.quantile(
        id_confs,
        torch.clamp(
            torch.tensor([tau_quantile-(1/40)*i, tau_quantile+(1/40)*i]),
            min=0, max=1
        ).to(torch.float64)
    ).tolist()
    for i in range(20)
]

# all to second
exit_ts.append(
    torch.quantile(
        id_confs,
        torch.tensor([0,1]).to(torch.float64)
    ).tolist()
)


window_strat_kwargs = [
    {
        "metric_name": "confidence",
        "window": w,
        "unc_flip":False,
        "tau":tau
    }
    for w in exit_ts
]

t_quants = torch.linspace(0,1,11)
exit_ts = torch.quantile(
    id_confs, t_quants.to(torch.float64)
)
single_strat_kwargs = [
    {
        "metric_name":args.exit_metric,
        "threshold": t.item(),
        "unc_flip":False
    }
    for t in exit_ts
]

window_result_rows = []
single_result_rows = []
for strat in window_strat_kwargs:

    two_head_res = evaluate_early_exit(
        logits[1:], 
        id_data,
        window_threshold_strategy,
        macs_list[1:],
        strategy_kwargs=[strat]
    )
    window_result_rows.append(two_head_res)

for strat in single_strat_kwargs:
    two_head_res = evaluate_early_exit(
        logits[1:], 
        id_data,
        single_threshold_strategy,
        macs_list[1:],
        strategy_kwargs=[strat]
    )
    single_result_rows.append(two_head_res)

single = pd.DataFrame(single_result_rows)
window = pd.DataFrame(window_result_rows)



def get_df_key(taskname, id_dataset, ood_dataset=None, unc="confidence"):

    if "risk" in taskname:
        if ood_dataset is not None:
            return f"{id_dataset} + {ood_dataset} {taskname} {unc}"
        else:
            return f"{id_dataset} {taskname} {unc}"
    elif "cov" in taskname:
        return f"{id_dataset} {taskname} {unc}"
    elif "FPR" in taskname:
        return f"OOD {ood_dataset} {taskname} {unc}"



def get_unc_y_label(taskname, id_dataset, ood_dataset=None, unc="confidence"):
    if "risk" in taskname:
        if ood_dataset is not None:
            return f"{EVAL_MAPPING[taskname]}"
        else:
            return f"{EVAL_MAPPING[taskname]}"
    elif "cov" in taskname:
        return f"{EVAL_MAPPING[taskname]}"
    elif "FPR" in taskname:
        return f"{EVAL_MAPPING[taskname]}"

df_key = get_df_key(
    args.unc_task, "imagenet", ood_dataset=None, unc=args.exit_metric
)

unc_y_label = get_unc_y_label(
    args.unc_task, "imagenet", ood_dataset=None, unc=args.exit_metric
)

dfs = [single, window]



lines = ["-",  "-"]
markers = ["o", "o"]
fig, ax = plt.subplots(2,2,figsize=(6,2.8), sharex=True, sharey="col")
row_text = ["single\nthreshold", "window\n(ours)"]
plt.minorticks_on()
for i, df in enumerate(dfs):
    ax[i][0].plot(
        df["imagenet_average_macs"],
        100-df["top1"],
        marker=markers[i],
        color="navy",
        label="ID acc.",
        linestyle=lines[i],
        alpha=0.6
    )
    ax[i][0].set_ylabel("top 1 acc. $\\rightarrow$", color="navy")
    ax[i][0].annotate(
        row_text[i] ,xy=(0, 0.5), xytext=(-ax[i][0].yaxis.labelpad - 15, 0),
                xycoords=ax[i][0].yaxis.label, textcoords='offset points',
                ha='center', va='center', rotation=90, size="large"
    )

    ax[i][1].plot(
        df["imagenet_average_macs"],
        np.stack(df[df_key])[:,0],
        marker=markers[i],
        color="green",
        label="ID MACs", linestyle=lines[i],
        alpha=0.6
    )


    ax[i][1].set_ylabel(unc_y_label, color="green")

    ax[i][0].grid(visible=True, which='minor', color='w', linewidth=0.5, axis="x")
    ax[i][1].grid(visible=True, which='minor', color='w', linewidth=0.5, axis="x")

ax[1][0].set_xlabel("MACs")
ax[1][1].set_xlabel("MACs")


fig.tight_layout(pad=0.2,w_pad=1.0, h_pad=1.0)

 # specify filename
spec = get_filename(config, seed=None)
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
filename = get_filename(config, seed=None) +  \
    f"_cascade_strat_comp_acc_{args.unc_task}_{args.exit_metric}_macs.pdf"
path = os.path.join(save_dir, filename)
fig.savefig(path)
print(f"figure saved to:\n{path}")




# 3 Exits
# ==============================================================================

# MSP
id_confs = [id_uncs[i]["confidence"] for i in range(len(id_uncs))]

# first 2 exits, corresponding to head1 and head3
strat_kwargs_list = []
# fix 2nd exit to 20% either side of tau2
# widen first exit symmetrically
quantiles = np.array([
        [70,20],
        [60,20],
        [50,20],
        [40,20],
        [30,20],
        [20,20],
])/100


for i in range(len(quantiles)):
    window_strat_kwargs = []
    for j in range(2):
        # get windows
        curve_reqs = {
            "cov@5":{"curve":"RC", "requirement": ["risk", 0.05]}, 
            "cov@10":{"curve":"RC", "requirement": ["risk", 0.10]}, 
            "risk@50":{"curve":"RC", "requirement": ["coverage", 0.5]}, 
            "risk@80": {"curve":"RC", "requirement":["coverage", 0.8]}, 
            "FPR@80":{"curve":"ROC", "requirement": ["tpr", 0.8]}, 
            "FPR@95": {"curve":"ROC", "requirement":["tpr", 0.95]}
        }

        curve_req = curve_reqs[args.unc_task]
        if curve_req["curve"] == "RC":
            tau = get_threshold(
                val_correct_idx[j], id_confs[j], 
                curve="RC", requirement=curve_req["requirement"]
            )

        tau_quantile = percentileofscore(id_confs[j], tau)/100.0
        exit_ts = torch.quantile(
            id_confs[j],
            torch.clamp(
                torch.tensor(
                    [
                        tau_quantile-quantiles[i,j]/2, 
                        tau_quantile+quantiles[i,j]/2
                    ]
                ),
                min=0, max=1
            ).to(torch.float64)
        ).tolist()


        window_strat_kwargs.append(
            {
                "metric_name": "confidence",
                "window": exit_ts,
                "unc_flip":False,
                "tau":tau
            }
        )
    strat_kwargs_list.append(window_strat_kwargs)

# so we get performance of exits by themselves
guaranteed_exit_window = np.array([
    [[0,0],[0,0]],
    [[0,1],[0,0]],
    [[0,1],[0,1]],
])
        
for i in range(3):
    window_strat_kwargs = []
    for j in range(2):
    
        window_strat_kwargs.append(
            {
                "metric_name": "confidence",
                "window": guaranteed_exit_window[i,j],
                "unc_flip":False,
                "tau":tau
            }
        )
    strat_kwargs_list.append(window_strat_kwargs)



result_rows = []
for windows in strat_kwargs_list:
    three_exit_res = evaluate_early_exit(
            logits, 
            id_data,
            window_threshold_strategy,
            macs_list,
            strategy_kwargs=windows
        )
    result_rows.append(three_exit_res)


adaptive_res = pd.DataFrame(result_rows[:len(quantiles)])
exit_res = pd.DataFrame(result_rows[len(quantiles):])


exit_macs = macs_list
adaptive_macs = adaptive_res["imagenet_average_macs"]

exit_perf = exit_res[df_key].apply(lambda x: x[0])
adaptive_perf = adaptive_res[df_key].apply(lambda x: x[0])


fig, ax = plt.subplots(1,1,figsize=(5,2))
markers = ["o", "+"]
ax.plot(exit_macs, exit_perf, marker=markers[0], color="black", label="MSDNet\nexits {2,3,5}")
ax.plot(
    adaptive_macs, adaptive_perf, 
    marker=markers[1], color="green", label="window-based\nearly exit"
)

ax.set_xlabel("Average MACs")
ax.set_ylabel(unc_y_label)

ax.legend(bbox_to_anchor=(1.03, 0.5), loc="center left")
fig.tight_layout()
ax.grid(visible=True, which='minor', color='w', linewidth=0.5, axis="x")
spec = get_filename(config, seed=None)
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
filename = get_filename(config, seed=config["seed"]) +  \
    f"_cascade_earlyexit_{args.unc_task}_macs_{args.exit_metric}.pdf"
path = os.path.join(save_dir, filename)
fig.savefig(path)
print(f"figure saved to:\n{path}")
