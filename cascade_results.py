"""Given the single models have already run, use their saved logits to 
extract results for the whole ensemble."""

import torch
import os
import pandas as pd
import json
from copy import copy
from utils.eval_utils import *
from utils.data_utils import (
    Data,
    get_preprocessing_transforms,
)
from argparse import ArgumentParser
from utils.train_utils import get_filename
from utils.cascade_utils import STRAT_FUNCS
from models.model_utils import model_generator
from ptflops import get_model_complexity_info
from scipy.stats import percentileofscore



# argument parsing ------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)


parser.add_argument(
    "--seeds",
    default=None,
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
    "--strategy_func",
    type=str,
    default="single_threshold_strategy"
)

parser.add_argument(
    "--exit_metric",
    type=str,
    default="confidence",
    help="confidence score used for cascade exit"
)

parser.add_argument(
    "--unc_task",
    type=str,
    default="cov@10",
    help="task/threshold to target for window strategy",
    choices=["cov@5", "cov@10", "risk@50", "risk@80", "FPR@80", "FPR@95"]
)



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

# ood data truncation
if "ood_truncate" not in config["test_params"]:
    config["test_params"]["ood_truncate"] = False
ood_truncate = config["test_params"]["ood_truncate"]


assert args.strategy_func in STRAT_FUNCS



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
    results.update(strategy_kwargs)

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


    # OOD data detection ------------------------------------------------------
    if ood_data is not None and config["test_params"]["ood_data"]:
        ood_results = {}
        for data in ood_data:
            ood_metrics = [
                uncertainties(
                    logits[i][data.name]
                )
                for i in range(2)
            ]


            # probability space ensembling
            if args.ensemble:
                # if ensemble then average in the probability space
                ood_log_probs = [
                    logits[0][data.name],
                    (
                        (
                            logits[0][data.name].to(torch.float64).softmax(dim=-1) + \
                            logits[1][data.name].to(torch.float64).softmax(dim=-1) 
                        )/2
                    ).log().to(torch.float32)
                ]
                
            else:
                ood_log_probs = [logits[0][data.name], logits[1][data.name]]


            # uncertainties for both members
            if args.ensemble:
                # just average the uncertainty scores themselves
                av_unc_metrics = {
                    unc_name: (
                        ood_metrics[0][unc_name] + ood_metrics[1][unc_name]
                    )/2
                    for unc_name in ood_metrics[0]
                }

                ood_metrics[1] = av_unc_metrics

            add_adjusted_uncs(ood_metrics,ood_log_probs)


            # flops on OOD data
            ood_results[f"{data.name}_average_macs"] = strategy_func(
                ood_metrics[0],
                cascade_macs[0]*torch.ones(
                    len(logits[0][data.name])
                ),
                (cascade_macs[1]+cascade_macs[0])*torch.ones(
                    len(logits[0][data.name])
                ),
                **strategy_kwargs
            ).mean().item()

            final_ood_metrics = {
                unc_name: strategy_func(
                        ood_metrics[0],
                        ood_metrics[0][unc_name], ood_metrics[1][unc_name],
                    **strategy_kwargs
                    )
                for unc_name in ood_metrics[0]
            }

            
            # ID 0, OOD 1
            domain_labels = torch.cat(
                [
                    torch.zeros(len(logits[0][labelled_data.name])), 
                    torch.ones(len(logits[0][data.name]))
                ]
            )


            # average uncertainties
            res = {
                f"{data.name} {k}": v.mean().item()
                for k, v in final_ood_metrics.items()
            }
            ood_results.update(res)

            # combine ID and OOD
            combined_metrics = {
                unc_name: torch.cat(
                    [
                        final_id_metrics[unc_name], # ID first then OOD
                        final_ood_metrics[unc_name]
                    ]
                )
                for unc_name in final_ood_metrics.keys()
            }


            # OOD detection
            res = ood_detect_results(
                domain_labels, combined_metrics, mode="ROC"
            )

            res = {
                f"OOD {data.name} ROC " + k: v 
                for k, v in res.items() 
                if k != "mode"
            }
            ood_results.update(res)

            res = ood_detect_results(
                domain_labels, combined_metrics, mode="FPR@95"
            )

            res = {
                f"OOD {data.name} FPR@95 " + k: v 
                for k, v in res.items() 
                if k != "mode"
            }
            ood_results.update(res)

            res = ood_detect_results(
                domain_labels, combined_metrics, mode="FPR@80"
            )

            res = {
                f"OOD {data.name} FPR@80 " + k: v
                for k, v in res.items()
                if k != "mode"
            }
            ood_results.update(res)


            # now we treat only correct classifications as positive
            # remove misclassifications from binary separation entirely
            # ID 0, OOD 1


            # SCOD
            # coverages are set over only ID data
            # mixing set at 1/2 ID 1/2 OOD
            for eval_mode in ["AURC", "risk@50", "risk@80"]:
                res = scod_results(
                    final_id_metrics, correct_idx, mode=eval_mode,
                    ood_metrics=final_ood_metrics
                )

                res = {
                    f"{labelled_data.name} + {data.name} {eval_mode} " + k: v
                    for k, v in res.items()
                    if k != "mode"
                }
                ood_results.update(res)

            # adjusted window to guard against distribution shift
            # done in an "offline processing" scenario
            # 20th percentile around tau (from val set) on test data
            # assume same number of OOD data as ID data
            # we have access to test uncertainties but not any labels
            if strategy_func.__name__ == "window_threshold_strategy":
                mixed_metrics = get_id_ood_mixture(id_metrics[0], ood_metrics[0])
                mixed_unc = mixed_metrics[strategy_kwargs["metric_name"]]
                mixed_confs = mixed_unc if "confidence" in strategy_kwargs["metric_name"] else -mixed_unc
                tau = strategy_kwargs["tau"]
                tau_quantile = percentileofscore(mixed_confs, tau)/100.0
 
                adjusted_window = torch.quantile(
                    mixed_confs,
                    torch.clamp(
                        torch.tensor([tau_quantile-(1/10), tau_quantile+(1/10)]),
                        min=0, max=1
                    ).to(torch.float64)
                ).tolist()

                new_strat_kwargs = copy(strategy_kwargs)
                new_strat_kwargs["window"] = adjusted_window

                final_id_metrics = {
                    unc_name: strategy_func(
                        id_metrics[0],
                        id_metrics[0][unc_name], id_metrics[1][unc_name],
                        **new_strat_kwargs
                    )
                    for unc_name in id_metrics[0]
                }

                final_ood_metrics = {
                    unc_name: strategy_func(
                            ood_metrics[0],
                            ood_metrics[0][unc_name], ood_metrics[1][unc_name],
                        **new_strat_kwargs
                        )
                    for unc_name in ood_metrics[0]
                }

                if f"{labelled_data.name}_average_macs_adj" not in results:

                    results[f"{labelled_data.name}_average_macs_adj"] = strategy_func(
                        id_metrics[0],
                        cascade_macs[0]*torch.ones(
                            len(logits[0][labelled_data.name])
                        ),
                        (cascade_macs[1]+cascade_macs[0])*torch.ones(
                            len(logits[0][labelled_data.name])
                        ),
                        **new_strat_kwargs
                    ).mean().item()

                # flops on OOD data
                ood_results[f"{data.name}_average_macs_adj"] = strategy_func(
                    ood_metrics[0],
                    cascade_macs[0]*torch.ones(
                        len(logits[0][data.name])
                    ),
                    (cascade_macs[1]+cascade_macs[0])*torch.ones(
                        len(logits[0][data.name])
                    ),
                    **new_strat_kwargs
                ).mean().item()

                # combine ID and OOD
                combined_metrics = {
                    unc_name: torch.cat(
                        [
                            final_id_metrics[unc_name], # ID first then OOD
                            final_ood_metrics[unc_name]
                        ]
                    )
                    for unc_name in final_ood_metrics.keys()
                }


                # OOD detection
                res = ood_detect_results(
                    domain_labels, combined_metrics, mode="ROC"
                )

                res = {
                    f"OOD {data.name} ROC " + k + " adj": v 
                    for k, v in res.items() 
                    if k != "mode"
                }
                ood_results.update(res)

                res = ood_detect_results(
                    domain_labels, combined_metrics, mode="FPR@95"
                )

                res = {
                    f"OOD {data.name} FPR@95 " + k + " adj": v 
                    for k, v in res.items() 
                    if k != "mode"
                }
                ood_results.update(res)

                res = ood_detect_results(
                    domain_labels, combined_metrics, mode="FPR@80"
                )

                res = {
                    f"OOD {data.name} FPR@80 " + k + " adj": v
                    for k, v in res.items()
                    if k != "mode"
                }
                ood_results.update(res)


                # now we treat only correct classifications as positive
                # remove misclassifications from binary separation entirely
                # ID 0, OOD 1


                # SCOD
                # coverages are set on ID data, since no access to OOD 
                # mixing set at 1/2 ID 1/2 OOD
                for eval_mode in ["AURC", "risk@50", "risk@80"]:
                    res = scod_results(
                        final_id_metrics, correct_idx, mode=eval_mode,
                        ood_metrics=final_ood_metrics
                    )

                    res = {
                        f"{labelled_data.name} + {data.name} {eval_mode} " + k + " adj": v
                        for k, v in res.items()
                        if k != "mode"
                    }
                    ood_results.update(res)
                     
         
        results.update(ood_results)


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


    strat = STRAT_FUNCS[args.strategy_func]

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
    id_uncs = id_uncs[0][args.exit_metric]
    # make into confidence
    # confidence is alias for MSP, is saved as confidence score for
    # legacy reasons
    id_confs = id_uncs if "confidence" in args.exit_metric else -id_uncs
    flip = True if "confidence" not in args.exit_metric else False
    if args.strategy_func in [
        "single_threshold_strategy"
    ]:
        # exit 80%, pass on 20%
        t_quants = torch.tensor([0.2])
        exit_ts = torch.quantile(
            id_confs, t_quants.to(torch.float64)
        )
        strat_kwargs = [
            {
                "metric_name":args.exit_metric,
                "threshold": t.item(),
                "unc_flip":flip
            }
            for t in exit_ts
        ]

    elif args.strategy_func == "window_threshold_strategy":

        curve_reqs = {
            "cov@5":{"curve":"RC", "requirement": ["risk", 0.05]}, 
            "cov@10":{"curve":"RC", "requirement": ["risk", 0.10]}, 
            "risk@50":{"curve":"RC", "requirement": ["coverage", 0.5]}, 
            "risk@80": {"curve":"RC", "requirement":["coverage", 0.8]}, 
            "FPR@80":{"curve":"ROC", "requirement": ["tpr", 0.8]}, 
            "FPR@95": {"curve":"ROC", "requirement":["tpr", 0.95]}
        }

        curve_req = curve_reqs[args.unc_task]

        # get tau on validation set
        if curve_req["curve"] == "RC":
            tau = get_threshold(
                val_correct_idx[0], id_confs, 
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
        print(f"target quantile: {tau_quantile}")

        # 10 percent either side of tau
        exit_ts = [
            torch.quantile(
                id_confs,
                torch.clamp(
                    torch.tensor([tau_quantile-(1/10), tau_quantile+(1/10)]),
                    min=0, max=1
                ).to(torch.float64)
            ).tolist()
        ]

        strat_kwargs = [
            {
                "metric_name": args.exit_metric,
                "window": w,
                "unc_flip":flip,
                "tau":tau
            }
            for w in exit_ts
        ]


    # single model
    print("single model")
    result_rows.append(
        evaluate_casade(
            logits,
            id_data,
            STRAT_FUNCS["single_model"],
            cascade_macs, cascade_params,
            ood_data=ood_data,
            strategy_kwargs={},
        )
    )

    # iterate through different levels of the strategy
    print("cascades")
    print(f"exit metric:{args.exit_metric}\nthresholds:{exit_ts}")
    for kwargs in strat_kwargs:
        print(f"exit strat params: {kwargs}")
        result_rows.append(
            evaluate_casade(
                logits, 
                id_data,
                strat, 
                cascade_macs, cascade_params,
                ood_data=ood_data,
                strategy_kwargs=kwargs,
            )
        )

    # ensemble
    print("second_model")
    result_rows.append(
        evaluate_casade(
            logits,
            id_data,
            STRAT_FUNCS["second_model"],
            cascade_macs, cascade_params,
            ood_data=ood_data,
            strategy_kwargs={},
        )
    )

    # results into DataFrame
    result_df = pd.DataFrame(result_rows)

    # save to subfolder with dataset and architecture in name
    # filename will have seed
    if config["test_params"]["results_save"]:
        spec = get_filename(config, seed=None)
        filename = get_filename(config) + "_cascade"
        save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        savepath = os.path.join(save_dir, f"{filename}_{seeds[0]}_{seeds[1]}_{strat.__name__}_{args.exit_metric}_{args.unc_task}_{args.suffix}.csv")

        # just overwrite what's there
        result_df.to_csv(savepath, mode="w", header=True)

        print(f"cascasde results saved to: {savepath}")
