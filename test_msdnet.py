"""Extract outputs on the test set for MSDNet.
"""
import torch

import os
import json 
import pandas as pd



from models.model_utils import (
    model_generator, 
    load_weights_from_file,
    get_macs_msdnet
)


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
    "--seed",
    default=None,
    type=int,
    help="random seed, can be specified as an arg or in the config."
)
parser.add_argument(
    "--weights_path",
    type=str,
    default=None,
    help="Optional path to weights, overrides config."
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



args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)

# set random seed
# prioritize arg seed
if args.seed is not None:
    torch.manual_seed(args.seed)
    # add seed into config dictionary
    config["seed"] = args.seed
elif "seed" in config and type(config["seed"]) == int:
    torch.manual_seed(config['seed'])
else:
    torch.manual_seed(0)
    config["seed"] = 0




# set gpu
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        config["gpu_id"]
    ).replace("[", "").replace("]", "")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {dev}")

# ood data truncation
if "ood_truncate" not in config["test_params"]:
    config["test_params"]["ood_truncate"] = False
ood_truncate = config["test_params"]["ood_truncate"]

# data-------------------------------------------------------------------------

if args.weights_path is not None and config["model"]["model_type"] == "msdnet":
    idx_path = os.path.join(
        os.path.split(args.weights_path)[0],
        "index.pth"
    )
    indices = torch.load(idx_path)
else:
    indices = None
id_data = Data(
    **config["id_dataset"],
    test_only=False,
    transforms=get_preprocessing_transforms(
        config["id_dataset"]["name"], resolution=224
    ),
    fast=False,
    indices=indices
)

test_loader = id_data.test_loader



# print transforms
print("="*80)
print(id_data.name)
print(id_data.test_set.transforms)
print("="*80)


# helper functions ------------------------------------------------------------

def get_outputs(model, loader, dev="cuda"):
    """Get the model outputs for a dataloader."""
    model.eval()
    # get ID data
    logit_dict = {head_name: [] for head_name in model.head_locs}
    feature_dict = {head_name: [] for head_name in model.head_locs}
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            labels, inputs = labels.to(dev), inputs.to(dev)

            # can optionally return last hidden layer features
            outputs = model(
                inputs
            )

            for head in logit_dict:
                logit_dict[head].append(outputs[head].to("cpu"))

    # convert lists of batch tensors into one long tensor
    logit_dict = {
        head: torch.cat(logits, dim=0) 
        for head, logits in logit_dict.items()
    }
  
    return logit_dict


def evaluate(
    model, id_data, ood_data=None, dev="cuda", shifted=False
):
    """Perform inference across datasets."""

    logits_dict = {}
    features_dict ={}
    print(f"eval on: {id_data.name}")

    # gradnorm uses the features from the final hidden layer

    logits = get_outputs(
        model, id_data.test_loader, dev=dev
    )

    # store logits for later
    logits_dict[f"{id_data.name}"] = logits



    # note that MSDNet uses its own indices from the public repo
    logits = get_outputs(
        model, id_data.val_loader, dev=dev
    )

    # store logits for later
    logits_dict[f"{id_data.name}_val"] = logits
   
    val_labels = torch.tensor(id_data.val_set.targets)
    val_preds = logits["head3"].argmax(dim=-1)
    print(
        f"accuracy: {(val_labels==val_preds).to(float).mean().item()}"
    )

    return logits_dict, features_dict


# evaluation-------------------------------------------------------------------

# load floating point densenet model and evaluate
model = model_generator(
    config["model"]["model_type"],
    **config["model"]["model_params"]
)


# try and get weights 
if args.weights_path is not None:
    weights_path = args.weights_path
elif (
    "weights_path" in config["model"]
    and
    config["model"]["weights_path"] is not None
):
    # where pretrained weights are
    weights_path = os.path.join(
        config["model"]["weights_path"],
        get_filename(config, seed=config["seed"]) + ".pth"
    )
# try:
print(f"Trying to load weights from: {weights_path}\n")
load_weights_from_file(model, weights_path)
print("Loading successful")


model.to(dev)

input_shape = (3,224,224)

macs_dict = get_macs_msdnet(
    model, config["model"]["model_type"], config["model"]["model_params"],
    input_shape
)

results = [macs_dict]

df = pd.DataFrame(results, index=["MACs"])


spec = get_filename(config, seed=None)
filename = get_filename(config) + "_comp"
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
savepath = os.path.join(save_dir, f"{filename}{args.suffix}.csv")

# just overwrite what's there
df.to_csv(savepath, mode="w", header=True)




# list of results dictionaries
result_rows = []

# eval floating point model
fp_logits, _ = evaluate(
    model, id_data, ood_data=None
)



print(f"datasets: {fp_logits.keys()}")


# stored for later use
precision_logit_dict = {}
precision_logit_dict["afp, wfp"] = fp_logits


# save the logits from all precisions
if config["test_params"]["logits_save"]:
    spec = get_filename(config, seed=None)
    filename = get_filename(config, seed=config["seed"])
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    savepath = os.path.join(save_dir, f"{filename}_logits{args.suffix}.pth")
    torch.save(precision_logit_dict, savepath)
    print(f"logits saved to: {savepath}")





