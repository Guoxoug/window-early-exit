
import torch
import torch.nn as nn
import torchvision as tv
import os
import json 
import pandas as pd
from models.model_utils import *
from utils.eval_utils import *
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
# bit of a hack to get around converting json syntax 
# deals with a list of integer ids
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

# load model -----------------------------------------------------------------

model = model_generator(
    config["model"]["model_type"],
    **config["model"]["model_params"]
)


# try and get weights 
# manual argument passed to script
if args.weights_path is not None:
    weights_path = args.weights_path

# automatically looks for weights according to config
elif (
    "weights_path" in config["model"]
    and
    config["model"]["weights_path"] is not None
):
    # where trained weights are
    weights_path = os.path.join(
        config["model"]["weights_path"],
        get_filename(config, seed=config["seed"]) + ".pth"
    )

print(f"Trying to load weights from: {weights_path}\n")
load_weights_from_file(model, weights_path)
print("Loading successful")

model.to(dev)





# data-------------------------------------------------------------------------

# resolution important as it is scaled differently for different models
trans = get_preprocessing_transforms(
    config["id_dataset"]["name"],
    resolution = model.resolution 
)

id_data = Data(
    **config["id_dataset"],
    test_only=False,
    transforms=trans,
    fast=False
)

test_loader = id_data.test_loader

# ood_data
# get id dataset normalisation values
if "ood_datasets" in config:
    ood_data = [
        Data(
            **ood_config,
            transforms=get_preprocessing_transforms(
                ood_config["name"],
                id_dataset_name=config["id_dataset"]["name"],
                resolution=model.resolution
            )
        )
        for ood_config in config["ood_datasets"]
    ]
else:
    ood_data = None


# print transforms
print("="*80)
print(id_data.name)
print(id_data.test_set.transforms)
print("="*80)
for data in ood_data:
    print("="*80)
    print(data.name)
    try:
        print(data.test_set.dataset.transforms)
    except:
        print(data.test_set.transforms)
    print("="*80)



# helper functions ------------------------------------------------------------

def get_logits_labels(
    model, loader, 
    dev="cuda", 
    early_stop=None # stop eval early 
):
    """Get the model outputs for a dataloader."""

    model.eval()
    # get ID data
    label_list = []
    logit_list = []
    feature_list = []
    count = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(loader)):
            labels, inputs = labels.to(dev), inputs.to(dev)
            batch_size = inputs.shape[0]
            # can optionally return last hidden layer features

            outputs = model(
                inputs,
            )
            label_list.append(labels.to("cpu"))
            logit_list.append(outputs.to("cpu"))

            count += batch_size
            if (
                early_stop is not None 
                and 
                count >= early_stop
            ):
                break

    logits, labels = torch.cat(logit_list, dim=0), torch.cat(label_list, dim=0)
    # clip to exactly match the early stop
    if early_stop is not None:
        logits, labels = logits[:early_stop], labels[:early_stop]

    return logits, labels



def evaluate(
    model, id_data, 
    ood_data=None, dev="cuda"
):

    logits_dict = {}
    print(f"eval on: {id_data.name}")

    logits, labels = get_logits_labels(
        model, id_data.test_loader, dev=dev,
    )

    # store logits for later
    logits_dict[f"{id_data.name}"] = logits.to("cpu")

    # validation logits
    logits, labels = get_logits_labels(
        model, id_data.val_loader, dev=dev,
    )
    logits_dict[f"{id_data.name}_val"] = logits.to("cpu")

    # OOD data stuff
    if ood_data is not None and config["test_params"]["ood_data"]:
        ood_results = {}
        for data in ood_data:
            print(f"eval on: {data.name}")

            ood_logits, _ = get_logits_labels(
                model, data.test_loader, dev=dev,
            )

            # balance the #samples between OOD and ID data
            # unless OOD dataset is smaller than ID, then it will stay smaller
            # this does not happen by default
            if ood_truncate:
                ood_logits = ood_logits[:len(logits)]
           
            logits_dict[f"{data.name}"] = ood_logits
            
         
    
    return logits_dict


# evaluation-------------------------------------------------------------------



# list of results dictionaries
result_rows = []

# eval floating point model
logits= evaluate(
    model, id_data, ood_data=ood_data
)

print(f"datasets: {logits.keys()}")



# stored for later use
# precision here due to legacy reasons
precision_logit_dict = {}
precision_logit_dict["afp, wfp"] = logits


# save the logits
spec = get_filename(config, seed=None)
filename = get_filename(config, seed=config["seed"])
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
savepath = os.path.join(save_dir, f"{filename}_logits{args.suffix}.pth")
torch.save(precision_logit_dict, savepath)
print(f"logits saved to {savepath}")


    

