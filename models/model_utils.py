# debugging
if __name__ == "__main__":
    from mobilenet_v2 import MobileNetV2
    from msdnet import MSDNet
    from efficientnet import EfficientNet, MBConvConfig
else:
    from models.mobilenet_v2 import MobileNetV2
    from models.msdnet import MSDNet
    from models.efficientnet import EfficientNet, MBConvConfig

from functools import partial
from ptflops import get_model_complexity_info
from copy import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Optional

MODEL_TYPES = [
    "mobilenetv2"
    "msdnet",
    "efficientnet", 
    "efficientnetb0", "efficientnetb1", 
    "efficientnetb2", "efficientnetb3", "efficientnetb4",
    "efficientnetb5",
    "mobilenetv2-1.0-160", "mobilenetv2-1.0-192", "mobilenetv2-1.0-224",
    "mobilenetv2-1.3-224",
    "mobilenetv2-1.4-224",
    "msdnet"
]
MODEL_TYPE_MAPPINGS = {
    "mobilenetv2": MobileNetV2,
    "mobilenetv2-1.0-160": MobileNetV2, 
    "mobilenetv2-1.0-192": MobileNetV2, 
    "mobilenetv2-1.0-224": MobileNetV2,
    "mobilenetv2-1.3-224": MobileNetV2,
    "mobilenetv2-1.4-224": MobileNetV2,
    "msdnet": MSDNet,
    "efficientnet":EfficientNet,
    "efficientnetb0":EfficientNet,
    "efficientnetb1":EfficientNet,
    "efficientnetb2":EfficientNet,
    "efficientnetb3":EfficientNet,
    "efficientnetb4":EfficientNet,
    "efficientnetb5":EfficientNet
}
MODEL_NAME_MAPPING = {
    "mobilenetv2": "MobileNetV2",
    "msdnet": "MSDNet",
    "efficientnet": "EfficientNet",
    "efficientnetb0": "EfficientNet-B0",
    "efficientnetb1": "EfficientNet-B1",
    "efficientnetb2": "EfficientNet-B2",
    "efficientnetb3": "EfficientNet-B3",
    "efficientnetb4": "EfficientNet-B4",
    "mobilenetv2-1.0-160": "MobileNetV2-1.0-160",
    "mobilenetv2-1.0-192": "MobileNetV2-1.0-192",
    "mobilenetv2-1.0-224": "MobileNetV2-1.0-224",
    "mobilenetv2-1.3-224": "MobileNetV2-1.3-224",
    "mobilenetv2-1.4-224": "MobileNetV2-1.4-224",
}

def efficientnet_config(
    arch
):
    """Get model config to load a specific efficientnet arch."""
    width_mult = {
        "efficientnetb0": 1.0,
        "efficientnetb1": 1.0,
        "efficientnetb2": 1.1,
        "efficientnetb3": 1.2,
        "efficientnetb4": 1.4,
        "efficientnetb5": 1.6,
    }
    depth_mult = {
        "efficientnetb0": 1.0,
        "efficientnetb1": 1.1,
        "efficientnetb2": 1.2,
        "efficientnetb3": 1.4,
        "efficientnetb4": 1.8,
        "efficientnetb5": 2.2,
    }
    dropout = {
        "efficientnetb0": 0.2,
        "efficientnetb1": 0.2,
        "efficientnetb2": 0.3,
        "efficientnetb3": 0.3,
        "efficientnetb4": 0.4,
        "efficientnetb5": 0.4,
    }
    resolution = {
        "efficientnetb0": 224,
        "efficientnetb1": 240,
        "efficientnetb2": 260,
        "efficientnetb3": 300,
        "efficientnetb4": 380,
        "efficientnetb5": 456,
    }
    bneck_conf = partial(
        MBConvConfig, 
        width_mult=width_mult[arch], 
        depth_mult=depth_mult[arch]
    )
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]

    return inverted_residual_setting, dropout[arch], resolution[arch]

def model_generator(model_type:str, **model_params) -> torch.nn.Module:
    """Construct a model following the supplied parameters."""
    assert model_type in MODEL_TYPES, (
        f"model type not supported"
        f"needs to be in {MODEL_TYPES}"    
    )

    # select model class
    Model = MODEL_TYPE_MAPPINGS[model_type]


    if "efficientnetb" in model_type:
        inv_res_setting, drop_rate, resolution = efficientnet_config(
            model_type
        )
        model_params.update(
            {
                "inverted_residual_setting": inv_res_setting,
                "dropout": drop_rate,
                "resolution": resolution
            }
        )
    # generic unpacking of pararmeters, need to match config file with 
    # model definition
    if model_type == "mobilenetv2-1.0-160":
        model_params.update(
            {
                "resolution": 160,
                "width_mult": 1.0
            }
        )
    if model_type == "mobilenetv2-1.0-192":
        model_params.update(
            {
                "resolution": 192,
                "width_mult": 1.0
            }
        )
    if model_type == "mobilenetv2-1.0-224":
        model_params.update(
            {
                "resolution": 224,
                "width_mult": 1.0
            }
        )
    if model_type == "mobilenetv2-1.3-224":
        model_params.update(
            {
                "resolution": 224,
                "width_mult": 1.3
            }
        )

    if model_type == "mobilenetv2-1.4-224":
        model_params.update(
            {
                "resolution": 224,
                "width_mult": 1.4
            }
        )
    model = Model(**model_params)

    return model

def load_weights_from_file(
    model, weights_path, dev="cuda", keep_last_layer=True, 
    early_exit_backbone=False, lightning=False
):
    """Load parameters from a path of a file of a state_dict."""



    if not lightning:
        state_dict = torch.load(weights_path, map_location=dev)
    else:
        state_dict = torch.load(weights_path, map_location="cpu")

    # special case for pretrained torchvision model
    # they fudged their original state dict and didn't change it

    if type(model) == MSDNet and "pth.tar" in weights_path:
        new_state_dict = {}
        # state_dict is a checkpoint with optimiser states and stuff
        #  rather than just a state dict with weights
        for k, v in state_dict["state_dict"].items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return
   

    new_state_dict = OrderedDict()

    # data parallel trained models have module in state dict
    # prune this out of keys
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    # load params
    state_dict = new_state_dict
    if not keep_last_layer:

        # filter out final linear layer weights
        state_dict = {
            key: params for (key, params) in state_dict.items()
            if "classifier" not in key and "fc" not in key
        }
        model.load_state_dict(state_dict, strict=False)
    elif early_exit_backbone:
        model.load_backbone(state_dict)
    else:
        print("loading weights")
        model.load_state_dict(state_dict, strict=True)
       



def get_macs_msdnet(
    model: nn.Module, model_type: str, model_params, input_shape,
):
    macs_dict = {}

    params_list = []
    for i in range(model_params["nBlocks"]):
        params_copy = copy(model_params)
        params_copy["exit_num"] = i+1
        params_list.append(params_copy)
    eval_models = [
        model_generator(model_type, **params)
        for params in params_list
    ]

    for j, early_exit in enumerate(eval_models):
        macs, params = get_model_complexity_info(
            early_exit, input_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=True
        )

        macs_dict[f"head{j+1}"] = macs

    return macs_dict

