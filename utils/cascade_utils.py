import torch
import numpy as np


def single_threshold_strategy(
    metrics: dict,
    data1: torch.Tensor, data2: torch.Tensor,
    threshold=0.5,
    unc_flip=False,
    metric_name="confidence"
):
    """Return combined data of final cascade output.
    A simple strategy that exits if the first member has confidence
    below a specified threshold.
    """

    conf_scores = metrics[metric_name]
    # change from uncertainty to confidence
    if unc_flip:
        conf_scores = -conf_scores
    out = conf_scores > threshold  # exit at first member

    assert data1.shape == data2.shape
    target_shape = [conf_scores.shape[0]] + \
        [1 for i in range(len(data1.shape)-1)]
    out = out.view(target_shape)

    return data1 * out + data2 * ~out


def window_threshold_strategy(
    metrics: dict,
    data1: torch.Tensor, data2: torch.Tensor,
    window=[0, 0.5],
    unc_flip=False,
    metric_name="confidence",
    tau=None 
):
    """Return combined data of final cascade output.
    Go to ensemble if score lies within a window.
    """

    conf_scores = metrics[metric_name]
    # change from uncertainty to confidence
    if unc_flip:
        conf_scores = -conf_scores

    # exit at first member
    # logical or
    out = (conf_scores > window[1]) + (conf_scores < window[0])

    assert data1.shape == data2.shape
    # pad shape with ones
    target_shape = [conf_scores.shape[0]] + \
        [1 for i in range(len(data1.shape)-1)]
    out = out.view(target_shape)

    return data1 * out + data2 * ~out

def single_model(
    metrics: dict,
    data1: torch.Tensor, data2: torch.Tensor,
    metric_name="confidence"
):
    """Just return outputs from the first model."""
    return data1

def second_model(
    metrics: dict,
    data1: torch.Tensor, data2: torch.Tensor,
    metric_name="confidence"
):
    """Just return outputs from the second model."""
    return data2


def random_strategy(
    metrics: dict,
    data1: torch.Tensor, data2: torch.Tensor,
    metric_name="confidence"
):
    """Random selection baseline."""
    rand = torch.randint(0,2,(len(data1),))
    out = torch.where(rand==0, data1, data2)
    return out


STRAT_FUNCS = {
    "single_threshold_strategy": single_threshold_strategy,
    "window_threshold_strategy": window_threshold_strategy,
    "single_model": single_model,
    "second_model": second_model,
    "random_strategy": random_strategy,
}
