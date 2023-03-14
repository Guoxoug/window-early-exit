import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import auc
from scipy.special import softmax
import torch.nn.functional as F



class TopKError(nn.Module):
    """
    Calculate the top-k error rate of a model. 
    """
    def __init__(self, k=1, percent=True):
        super().__init__()
        self.k = k
        self.percent = percent 

    def forward(self, labels, outputs):
        # get rid of empty dimensions
        if type(labels) == np.ndarray:
            labels = torch.tensor(labels)
        if type(outputs) == np.ndarray:
            outputs = torch.tensor(outputs)
        labels, outputs = labels.squeeze(), outputs.squeeze()
        _, topk = outputs.topk(self.k, dim=-1)
        # same shape as topk with repeated values in class dim
        labels = labels.unsqueeze(-1).expand_as(topk)
        acc = torch.eq(labels, topk).float().sum(dim=-1).mean()
        err = 1 - acc
        err = 100 * err if self.percent else err
        return err.item()
        


# printing --------------------------------------------------------------------

def print_results(results: dict):
    """Print the results in a results dictionary."""
    print("="*80)
    for k, v in results.items():
        if type(v) == float:
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
    print("="*80)




import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics._ranking import _binary_clf_curve
import warnings
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import torchvision as tv


DOUBLE_INFO = torch.finfo(torch.double)
METRIC_NAME_MAPPING = {
    "confidence": "MSP",
    "entropy": "$-\mathcal{H}$",
    "max_logit": "Max Logit",
    "energy": "Energy"

}

SOFTMAX_METRICS = ["confidence", "entropy"]

def get_metric_name(unc):
    if unc in METRIC_NAME_MAPPING:
        return METRIC_NAME_MAPPING[unc]
    else:
        return unc


def entropy(probs: torch.Tensor, dim=-1):
    "Calcuate the entropy of a categorical probability distribution."
    log_probs = probs.log()
    ent = (-probs*log_probs).sum(dim=dim)
    return ent




def uncertainties(
    logits: torch.Tensor
) -> dict:
    """Calculate uncertainty measures from categorical output."""

    # increase precision
    # softmax compresses high probability values close together
    logits = logits.type(torch.DoubleTensor)


    probs = logits.softmax(dim=-1)
    max_logit = -logits.max(dim=-1).values
    conf = probs.max(dim=-1).values # MSP
    ent = entropy(probs, dim=-1)
    energy = -torch.logsumexp(logits, dim=-1)

    # IMPORTANT
    # Max Softmax Probability is aliased as confidence in this code
    uncertainty = {
        'confidence': conf, 
        'entropy': ent, 
        "max_logit": max_logit,
        "energy": energy,
    }

    return uncertainty


def get_ood_metrics_from_combined(metrics, domain_labels):
    """Extract metrics only related to OOD data from combined data."""
    OOD_metrics = {}
    for key, metric in metrics.items():
        OOD_metrics[key] = metric[domain_labels == 1]

    return OOD_metrics


# OOD detection ---------------------------------------------------------------

def fpr_at_recall(labels, scores, recall_level):
    """Get the false positive rate at a specific recall."""
    # recall and TPR are the same thing

    # positive is ID now
    labels = ~labels.astype(bool)
    scores = -scores
    precision, recall, thresholds = precision_recall_curve(
            labels, scores
    )

    # postive if >= threshold, recall and precision have an extra value
    # for 0 recall (all data classified as negative) at the very end
    # get threshold closest to specified (e.g.95%) recall
    cut_off = np.argmin(np.abs(recall-recall_level))
    t = thresholds[cut_off]


    negatives = ~labels 

    # get positively classified samples and filter
    fps = np.sum(negatives * (scores >= t))

    return fps/np.sum(negatives)


def ood_detect_results(
    domain_labels,
    metrics,
    mode="ROC",
    classes_flipped=None,
):
    """Evaluate OOD data detection using different uncertainty metrics."""

    # iterate over different metrics (e.g. mutual information)
    assert mode in ["PR", "ROC", "FPR@95", "FPR@80"]
    domain_labels = np.asarray(domain_labels)
    results = {"mode": mode}
    for key in metrics.keys():
        pos_label = 1
        if 'confidence' in key:
            pos_label = 0

        results[key] = ood_detect(
            domain_labels,
            metrics[key],
            mode=mode,
            pos_label=pos_label
        )

    return results


def ood_detect(
    domain_labels,
    metric,
    mode,
    pos_label=1,
    labels=None,
    preds=None
):
    """Calculate the AUPR or AUROC for OOD detection (binary)."""
    scores = metric
    scores = np.asarray(scores, dtype=np.float128)
    if pos_label != 1:
        scores *= -1.0
        
    # if there is overflow just clip to highest float
    scores = np.nan_to_num(scores) 

    # receiver operating characteristic
    if mode == 'ROC':
        # symmetric so don't care  
        roc_auc = roc_auc_score(domain_labels, scores)
        # percent
        return roc_auc * 100

    elif mode == "FPR@95":
        recall_level = 0.95
        # note that the labels are ID 1, OOD 0
        # recall of ID is the same thing as TPR
        fpr = fpr_at_recall(domain_labels, scores, recall_level)
        # percent
        return fpr * 100
    
    elif mode == "FPR@80":
        recall_level = 0.8
        fpr = fpr_at_recall(domain_labels, scores, recall_level)
        # percent
        return fpr * 100



# SC and SCOD stuff ------------------------------------------------------------

def risk_coverage_curve(y_true, y_score, sample_weight=None):
    if sample_weight is None:
        sample_weight = 1
    sorted_idx = y_score.argsort(descending=True)
    # risk for each coverage value rather than recall
    # add one to cover situation with zero coverage, assume risk is zero
    # when nothing is selected
    coverage = torch.linspace(0, 1, len(y_score) + 1)
    # invert labels to get invalid predictions (miscls + OOD)
    sample_costs = ~(y_true.to(bool)) * sample_weight
    sorted_cost = sample_costs[sorted_idx]
    summed_cost = torch.cumsum(sorted_cost, 0)
    n_selected = torch.arange(1, len(y_score) + 1)
    # zero risk when none selected
    risk = torch.cat([torch.zeros(1), summed_cost/n_selected])
    thresholds = y_score[sorted_idx] # select >= threshold
    return risk, coverage, thresholds

def risk_recall_curve(y_true, y_score, pos_label=None, sample_weight=None):
    # see https://github.com/scikit-learn/scikit-learn/blob/80598905e/sklearn/metrics/_ranking.py
    
    # the sample weight is directly multiplied with each tps/fps
    # we use this to weigh misclassifications vs correct classifications
    weighted_fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )
    # unweighted tps and fps
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=None
    )
    # note fps[i] is the #fps with value >= thresholds[i]
    # thresholds is decreasing
    ps = tps + fps
    risk = np.divide(weighted_fps, ps, where=(ps != 0))

    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[-1] == 0 <=> y_true == all negative labels
    if tps[-1] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]
    # zero risk at zero recall
    return np.hstack((0,risk)), np.hstack((0, recall)), thresholds


def get_id_ood_mixture(id_metrics, ood_metrics, alpha=0.5):
    """Get a mixture of ID and OOD data with interpolation factor alpha.
    Doesn't give ID/OOD labels, 
    assume we don't have access to labels at deployment.
    """

    combined_metrics = {}
    for i, key in enumerate(id_metrics.keys()):
        id_unc = id_metrics[key]
        ood_unc = ood_metrics[key]
        if i == 0: # decide on subsampling on the first metric
            ratio = (1-alpha)/alpha
            
            # subsample randomly 
            # reset random state
            # note that this should only be called during evaluation
            torch.manual_seed(0)

            # subsample OOD data
            if ratio < len(ood_unc)/len(id_unc): # not enough ID data
                n_ood = round(len(id_unc) * ratio)
                shuffle_idx = torch.randperm(len(ood_unc))[:n_ood]
                subsample_id = False
                
            
            else: # not enough OOD data, subsample ID
                n_id = round(len(ood_unc)/ratio)
                shuffle_idx = torch.randperm(len(id_unc))[:n_id]
                subsample_id = True

        # for all metrics
        if subsample_id:
            id_unc = id_unc[shuffle_idx]
        else:
            ood_unc = ood_unc[shuffle_idx]

        combined_unc = torch.cat(
            [
                id_unc,
                ood_unc
            ]
        )

        combined_metrics[key] = combined_unc

    return combined_metrics
        


def scod_results(
    metrics,
    correct_idx,
    ood_metrics=None,
    alpha=0.5,
    beta=1.0,
    mode="AURC"
):
    """Get SCOD results for a single problem setting."""
    assert mode in [
        "AURC", "AURR", 
        "risk@95", # risk at recall
        "cov@5", "cov@10",
        "risk@80", "risk@50" # risk at coverage
    ]
    results = {"mode": mode}
    for i, key in enumerate(metrics.keys()):
        id_unc = metrics[key]
 
        if ood_metrics is not None:
            ood_unc = ood_metrics[key]
            if i == 0: # decide on subsampling on the first metric
                ratio = (1-alpha)/alpha
                
                # subsample randomly 
                # reset random state
                # note that this should only be called during evaluation
                torch.manual_seed(0)

                # subsample OOD data
                if ratio < len(ood_unc)/len(id_unc): # not enough ID data
                    n_ood = round(len(id_unc) * ratio)
                    shuffle_idx = torch.randperm(len(ood_unc))[:n_ood]
                    subsample_id = False
                    
                
                else: # not enough OOD data, subsample ID
                    n_id = round(len(ood_unc)/ratio)
                    shuffle_idx = torch.randperm(len(id_unc))[:n_id]
                    subsample_id = True
                    # shuffle_idx is a tensor of ints
                    # correct idx is a tensor of bools
                    correct_idx = correct_idx[shuffle_idx]


            # for all metrics
            if subsample_id:
                id_unc = id_unc[shuffle_idx]
            else:
                ood_unc = ood_unc[shuffle_idx]

            if i == 0:
                # different, now misclassification is always a cost of 1
                # by default assume that missed OOD sample is the same as 
                # an error
                ood_weight = beta
                sample_weights = torch.cat(
                    [
                        torch.ones(len(id_unc)),
                        torch.ones(len(ood_unc)) * ood_weight
                    ]
                )
                # ordering is (ID correct + ID incorrect) OOD

                # fix labels on the first metric as they are the same
                y_true = torch.cat(
                    [
                        correct_idx.to(int),  # bool to int, 1 for correct 0 for wrong
                        torch.zeros(len(ood_unc)) # zero for ood
                    ]
                )


            combined_unc = torch.cat(
                [
                    id_unc,
                    ood_unc
                ]
            )
            

        else:
            sample_weights=None
            y_true = correct_idx.to(int)
            combined_unc = id_unc


        # indices of ID for setting threshold
        # ID data concatenated before OOD data
        id_idx = torch.arange(len(id_unc))
        # change from unc to confidence scores
        scale = -1
        if 'confidence' in key:
            scale= 1

        results[key] = scod(
            y_true,
            scale*combined_unc,
            sample_weight=sample_weights,
            mode=mode,
            ood_metrics=ood_metrics,
            id_idx=id_idx
        )

    return results

def scod(
    y_true,
    metric,
    sample_weight=None,
    mode="AURC",
    ood_metrics=None,
    id_idx=None
):
    if mode in  [
        "AURC", 
        "cov@5", "cov@10",
        "risk@50", "risk@80"
    ]:
        risk, coverage, thresholds = risk_coverage_curve(
            y_true, metric, sample_weight=sample_weight
        )

        if mode == "AURC":
            # use trapezium rule here 
            aurc = torch.trapz(risk, coverage).item()
            return aurc
        elif "cov" in mode: # coverage at risk of 5, high safety requirement
            # low coverage risk values are unstable, so allow for some more samples
            
            if "@5" in mode:
                cut_off = np.argmin(np.abs(risk[500:]-0.05))+500
            elif "@10" in mode:
                cut_off = np.argmin(np.abs(risk[500:]-0.1))+500

            # sometimes happens if score is not useful for SC
            if cut_off >= len(thresholds):
                cut_off = len(thresholds) - 1
            return [100*coverage[cut_off].item(), thresholds[cut_off].item()]
        
        elif "risk" in mode: # this is risk at coverage
            # determine threshold on ID data, use on both ID and OOD
            if ood_metrics is not None:
                assert id_idx is not None
                id_risk, id_coverage, id_thresholds = risk_coverage_curve(
                    y_true[id_idx], metric[id_idx]
                )
                if "50" in mode:
                    id_cut_off = np.argmin(np.abs(id_coverage-0.5))
                elif "80" in mode:
                    id_cut_off = np.argmin(np.abs(id_coverage-0.8))
                thresh = id_thresholds[id_cut_off].item()
                cut_off = np.argmin(np.abs(thresholds-thresh))

            # no OOD
            else:
                if "50" in mode:
                    cut_off = np.argmin(np.abs(coverage-0.5))
                elif "80" in mode:
                    cut_off = np.argmin(np.abs(coverage-0.8))
            return [100*risk[cut_off].item(), thresholds[cut_off].item()]

    else:
        risk, recall, thresholds = risk_recall_curve(
            y_true, metric, sample_weight=sample_weight
        )
        if mode == "AURR":
            # auc is calculated as 
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
            aurr = np.sum(np.diff(recall) * np.array(risk)[1:])
            return aurr
        elif mode == "risk@95": # this is for 95 recall of correct classes
            cut_off = np.argmin(np.abs(recall-0.95))
            spec_risk = risk[cut_off]
            return spec_risk
        
def get_threshold(y_true, y_score, curve="RC", requirement=["risk", 0.05]):
    """Get an operating threshold for a certain curve given a required value"""

    if curve=="RC":
        risk, coverage, thresholds = risk_coverage_curve(
            y_true, y_score
        )
        if requirement[0] == "risk":
            cut_off = np.argmin(np.abs(risk[500:]-requirement[1]))+500
            print("coverage  ", coverage[cut_off].item())
            return thresholds[cut_off].item()
        else:
            cut_off = np.argmin(np.abs(coverage-requirement[1]))
            return thresholds[cut_off].item()
    elif curve=="ROC":
        fpr, tpr, thresholds = roc_curve(
            y_true, y_score
        )
        if requirement[0] == "fpr":
            cut_off = np.argmin(np.abs(fpr-requirement[1]))
            return thresholds[cut_off].item()
        else:
            cut_off = np.argmin(np.abs(tpr-requirement[1]))
            print("tpr ", tpr[cut_off])
            return thresholds[cut_off].item()



