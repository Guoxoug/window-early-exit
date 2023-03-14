"""Script reads all the json files and replaces results and data directories"""
import os
import json

# relevant paths
SAVED_MODELS = "models/saved_models" # keep this the same

# insert the absolute paths to your own directories
RESULTS = "/path/to/results"
if not os.path.exists(RESULTS):
    os.mkdir(RESULTS)
data_paths = {
    "imagenet" : "/path/to/imagenet", # various formats work

    # directory 1 level above the image directory
    # torch needs subdirectories for classes
    "openimage-o": "/path/to/openimage-o", # needs datalist in folder
    "inaturalist": "/path/to/iNaturalist",
}

# passed eventually to DataLoader class
# note that this is per-gpu for lightning
NUM_WORKERS = 4
current_dir = os.getcwd()
# get json files
json_files = [
    f for f in os.listdir(current_dir) 
    if f.endswith(".json")
]

for file in json_files:
    with open(file, "r+") as f:
        config = json.load(f)

        # update information
        # weights location
        config["model"]["weights_path"] = os.path.join(
            SAVED_MODELS, 
            config["model"]["model_type"] + "_" + config["id_dataset"]["name"]
        )

        # directory to save results
        config["test_params"]["results_savedir"] = RESULTS

        # datasets paths
        config["id_dataset"]["datapath"] = data_paths[
            config["id_dataset"]["name"]
        ]
        config["id_dataset"]["num_workers"] = NUM_WORKERS
        config["id_dataset"]["idx_path"] = "misc/imagenet_idx"

        for i, dataset in enumerate(config["ood_datasets"]):
            config["ood_datasets"][i]["datapath"] = data_paths[
                config["ood_datasets"][i]["name"]
            ]
            config["ood_datasets"][i]["num_workers"] = NUM_WORKERS

        # write to file
        f.seek(0)
        json.dump(config, f, indent=4)
        f.truncate()
