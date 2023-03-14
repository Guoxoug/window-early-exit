"""Given the single models have already run, use their saved logits to 
extract results for the whole ensemble."""

import torch
import os
import pandas as pd
import json
from copy import copy
from utils.eval_utils import *
from utils.data_utils import *
from utils.cascade_utils import *
from argparse import ArgumentParser
from utils.train_utils import get_filename
from utils.cascade_utils import *
from models.model_utils import model_generator
from ptflops import get_model_complexity_info
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()



# argument parsing ------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)


parser.add_argument(
    "--seeds",
    default="12",
    type=str,
    help="string containing random seeds of the two cascade members in order."
)

parser.add_argument(
    "--secondary_config_path",
    default=None,
    type=str,
    help="if cascade is heterogeneous then this points to the second model"
)



parser.add_argument(
    "--gpu",
    type=int,
    default=None,
    help="gpu override for debugging to set the gpu to use."
)

parser.add_argument(
    "--suffix",
    default="",
    help="a suffix to differentiate a file"
)

parser.add_argument(
    "--ensemble",
    default=1,
    help="whether to ensemble the models together on the second stage"
)


parser.add_argument(
    "--unc_task",
    type=str,
    default="cov@5",
    help="task/threshold to target for window strategy",
    choices=["cov@5", "cov@10", "risk@50", "risk@80", "FPR@80", "FPR@95"]
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

# ood data truncation
if "ood_truncate" not in config["test_params"]:
    config["test_params"]["ood_truncate"] = False
ood_truncate = config["test_params"]["ood_truncate"]

# determinism in testing
torch.backends.cudnn.benchmark = False




# evaluation functions --------------------------------------------------------


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

def add_adjusted_uncs(metrics_list, log_probs):
    """Change metrics list in place."""

    # softmax uncertainties where the ensemble unc is calc after
    # ensembling the probabilities of the softmax
    for i in range(2):
        ens_uncs = {
            f"{unc_name}_ens": unc_values
            for unc_name, unc_values in
            uncertainties(
                log_probs[i]  # will only calculate logit uncertainties
            ).items()
            if unc_name in SOFTMAX_METRICS
        }
        metrics_list[i].update(ens_uncs)


def evaluate_casade(
    logits, labelled_data, strategy_func,
    cascade_macs, cascade_params,
    ood_data=None,  data_name=None, 
    strategy_kwargs={}   
):
    """Evaluate the cascade's performance."""

    if data_name is None:
        data_name = labelled_data.name
    print(f"eval on: {data_name}")
    try:
        labels = torch.tensor(labelled_data.test_set.targets)
    except:
        labels = torch.tensor(id_data.test_set.targets)

    val_labels = torch.tensor(labelled_data.val_set.targets)


    # results =================================================================

    results = {}
    results["dataset"] = data_name
    results["strategy"] = strategy_func.__name__

    # ID stuff ----------------------------------------------------------------
    # get uncertainties for each member
    id_metrics = [
        uncertainties(
            logits[i][data_name],
        )
        for i in range(2)
    ]


    if args.ensemble:
        # if ensemble then average in the probability space
        log_probs = [
            logits[0][data_name],
            (
                (
                    logits[0][data_name].to(torch.float64).softmax(dim=-1) + \
                    logits[1][data_name].to(torch.float64).softmax(dim=-1) 
                )/2
            ).log().to(torch.float32)
        ]
        
    else:
        log_probs = [logits[0][data_name], logits[1][data_name]]


    

    # uncertainties for both members
    if args.ensemble:
        # just average the uncertainty scores themselves
        av_unc_metrics = {
            unc_name: (id_metrics[0][unc_name] + id_metrics[1][unc_name])/2
            for unc_name in id_metrics[0]
        }

        id_metrics[1] = av_unc_metrics

    # ensemble uncertainties i.e. not average but using average softmax 
    add_adjusted_uncs(id_metrics, log_probs)



    # comp cost 
    results["nparams"] = cascade_params[0] + cascade_params[1]
    results["lowest_macs"] = cascade_macs[0]
    results["highest_macs"] = cascade_macs[0] + cascade_macs[1]
    results[f"{labelled_data.name}_average_macs"] = strategy_func(
        id_metrics[0],
        cascade_macs[0]*torch.ones(len(logits[0][data_name])),
        (cascade_macs[1]+cascade_macs[0])*torch.ones(len(logits[0][data_name])),
        **strategy_kwargs
    ).mean().item()

    # id evaluation
    final_id_log_probs = strategy_func(
        id_metrics[0],
        log_probs[0], log_probs[1],
        **strategy_kwargs
    )
    results.update(
        eval_id_measures(final_id_log_probs, labels)
    )

    final_id_metrics = {
        unc_name: strategy_func(
            id_metrics[0],
            id_metrics[0][unc_name], id_metrics[1][unc_name],
            **strategy_kwargs
        )
        for unc_name in id_metrics[0]
    }

    res = {
        f"{data_name} {k}": v.mean().item()
        for k, v in final_id_metrics.items()
    }
    results.update(res)


    # AUROC for misclassification detection
    max_logits, preds = final_id_log_probs.max(dim=-1)
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

if __name__ == "__main__":

    # data---------------------------------------------------------------------

    id_data = Data(
        **config["id_dataset"],
        test_only=False,
        transforms=get_preprocessing_transforms(config["id_dataset"]["name"]),
        fast=False
    )

    test_loader = id_data.test_loader
    train_loader = id_data.train_loader  # for ptq calibration

    # ood_data
    # get id dataset normalisation values
    if "ood_datasets" in config:
        ood_data = [
            Data(
                **ood_config,
                transforms=get_preprocessing_transforms(
                    ood_config["name"],
                    id_dataset_name=config["id_dataset"]["name"]
                )
            )
            for ood_config in config["ood_datasets"]
        ]
    else:
        ood_data = None




    # load logits  ------------------------------------------------
    # for homogeneous seeds are 1,2
    if args.seeds is not None:
        seeds = list(args.seeds)
    elif args.secondary_config_path is not None:
        seeds = [1,1]
    else:
        seeds = [1,2] 

    # results path generated as results_savedir/arch_dataset
    results_paths = [
        os.path.join(
            configs[i]["test_params"]["results_savedir"],
            get_filename(configs[i], seed=None)
        )
        for i in range(2)
    ]   

    
    logits_paths = [
        os.path.join(
            results_paths[i], 
            get_filename(configs[i], seed=seeds[i]) + f"_logits.pth"
        )  # results_savedir/arch_dataset/arch_dataset_seed_logits.pth
        for i in range(2)
    ]

    # these are actually dictionaries
    # containing many difference quantization levels
    # just working with full precision for this project
    print("Loading logits")
    logits = [
        torch.load(path)["afp, wfp"] for path in logits_paths
    ]
    print("Loading complete")
    print(logits[0].keys())


    # get computational cost --------------------------------------------------
    # actual latency/throughput measured in other script

    # load base model
    model = model_generator(
        config["model"]["model_type"],
        **config["model"]["model_params"]
    )
    macs, params = get_model_complexity_info(
        model, (3, model.resolution, model.resolution), 
        as_strings=False, print_per_layer_stat=False
    )

    if args.secondary_config_path is not None:
        secondary_model = model_generator(
        secondary_config["model"]["model_type"],
        **secondary_config["model"]["model_params"]
        )
        secondary_macs,  secondary_params= get_model_complexity_info(
            secondary_model, 
            (3, secondary_model.resolution, secondary_model.resolution), 
            as_strings=False, print_per_layer_stat=False
        )

        cascade_macs = [macs, secondary_macs]
        cascade_params = [params, secondary_params]
    else:
        cascade_macs = [macs, macs]
        cascade_params = [params, params]



    # now evaluate on logits --------------------------------------------------
    result_rows = []
    
    val_logits = [logits[i][f"{id_data.name}_val"] for i in range(2)]
    val_labels = torch.tensor(id_data.val_set.targets)
    
    if args.ensemble:
        val_log_probs = [
            logits[0][f"{id_data.name}_val"],
            (
                (
                    logits[0][f"{id_data.name}_val"].to(torch.float64).softmax(dim=-1) +
                    logits[1][f"{id_data.name}_val"].to(torch.float64).softmax(dim=-1)
                )/2
            ).log().to(torch.float32)
        ]
    else:
        val_log_probs = [
            logits[0][f"{id_data.name}_val"], logits[1][f"{id_data.name}_val"]
        ]
    val_correct_idx = [
        val_labels == val_log_probs[i].argmax(dim=-1)
        for i in range(2)
    ]
    print(
        f"val acc: {val_correct_idx[0].sum().item()/len(val_labels)*100:.2f}"
    )

    id_uncs = [
        uncertainties(
            logits=val_logits[i],
        ) for i in range(2)
    ]

    add_adjusted_uncs(id_uncs, val_log_probs)
    id_uncs = id_uncs[0]["confidence"]
    # make into confidence
    # confidence is alias for MSP, is saved as confidence score for
    # legacy reasons
    id_confs = id_uncs if "confidence" in "confidence" else -id_uncs
    flip = True if "confidence" not in "confidence" else False

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

    tau_quantile = percentileofscore(id_confs, tau)/100.0
    exit_ts = [
        torch.quantile(
            id_confs,
            torch.clamp( # twenty steps, note that it claps after a certain point
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
            "metric_name":"confidence",
            "threshold": t.item(),
            "unc_flip":False
        }
        for t in exit_ts
    ]

    window_result_rows = []
    single_result_rows = []


    for strat in window_strat_kwargs:
        window_result_rows.append(
            evaluate_casade(
                logits, 
                id_data,
                window_threshold_strategy, 
                cascade_macs, cascade_params,
                strategy_kwargs=strat,
            )
        )
    for strat in single_strat_kwargs:
        single_result_rows.append(
            evaluate_casade(
                logits, 
                id_data,
                single_threshold_strategy, 
                cascade_macs, cascade_params,
                strategy_kwargs=strat,
            )
        )
    # iterate through different levels of the strategy

    single = pd.DataFrame(single_result_rows)
    window = pd.DataFrame(window_result_rows)

    def get_df_key(taskname, id_dataset, ood_dataset=None, unc="confidence"):
        unc = unc + "_ens"
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
        args.unc_task, "imagenet", ood_dataset=None, unc="confidence"
    )

    unc_y_label = get_unc_y_label(
        args.unc_task, "imagenet", ood_dataset=None, unc="confidence"
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
        f"_cascade_strat_comp_acc_{args.unc_task}_confidence_macs.pdf"
    path = os.path.join(save_dir, filename)
    fig.savefig(path)
    print(f"figure saved to:\n{path}")
