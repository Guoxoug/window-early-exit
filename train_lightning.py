from collections import OrderedDict
import os
import json
import signal
from pytorch_lightning.plugins.environments import SLURMEnvironment
import torch
import pytorch_lightning as pl
from utils.data_utils import (
    Data, get_preprocessing_transforms, TRAIN_DATASETS
)
from utils.eval_utils import TopKError
from utils.train_utils import (
    OPTIMIZER_MAPPING,
    SCHEDULER_MAPPING,
    get_filename,
    save_state_dict
)
from models.model_utils import (
     model_generator
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.profilers import AdvancedProfiler
from argparse import ArgumentParser

# argument parsing
parser = ArgumentParser()

parser.add_argument(
    "config_path",
    help="path to the experiment config file for this training script"
)

parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="random seed, can be specified as an arg or in the config."
)

parser.add_argument(
    "--gpu",
    type=str,
    default=None,
    help="gpu override for debugging to set the gpu to use."
)


parser.add_argument(
    "--amp",
    default=1,
    choices=[0, 1],
    help="whether to use mixed precision training",
    type=int
)

parser.add_argument(
    "--slurm",
    default=0,
    choices=[0, 1],
    help="whether training using slurm",
    type=int
)



args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)

# set gpu
# lightning takes gpus as a list
if args.gpu is not None:
    config["gpu_id"] = [int(idx) for idx in list(args.gpu)]
elif "gpu_id" in config and (
    type(config["gpu_id"]) == list
    or 
    type(config["gpu_id"]) == int
    or
    type(config["gpu_id"]) == int
):
    pass
else:
    config["gpu_id"] = [0]

print(f"gpus allowed to be used: ", config["gpu_id"])
# handled later on by lightning


assert config["id_dataset"]["name"] in TRAIN_DATASETS, "not valid train set"

print(f"gpus available to torch {torch.cuda.device_count()}")

# set random seed
# CL arg overrides value in config file
if args.seed is not None:
    pl.seed_everything(args.seed)

    # add seed into config dictionary
    config["seed"] = args.seed


elif "seed" in config and type(config["seed"]) == int:
    pl.seed_everything(config['seed'])


# no seed in config or as CL arg
else:
    pl.seed_everything(0)
    config["seed"] = 0




# set training device, defaults to cuda
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dev = torch.device("cpu")
print(f"using {dev} for training")
# multigpu
if (
    config["ddp"]
    and torch.cuda.device_count() > 1
    and dev.type == "cuda"
):

    multi_gpu = True
    print("using multi-gpu")
    # distributed data parallel
    strategy = DDPStrategy(find_unused_parameters=False)
    # effective batch size needs to be adjusted
    # divide by the number of gpus as there is one dataloader per gpu
    config["id_dataset"]["batch_size"] = int(
        config["id_dataset"]["batch_size"] / len(config["gpu_id"])
    )


else:
    multi_gpu = False
    print("using single gpu")
    strategy = None



# load the model -------------------------------------------------------------
model = model_generator(
    config["model"]["model_type"],
    **config["model"]["model_params"]
)



# training loss
# only standard one hot CE 
print("training with cross entropy")
criterion = torch.nn.CrossEntropyLoss()


# directory to save weights from training
if (
    "weights_path" in config["model"]
    and
    config["model"]["weights_path"] is not None
):
    # make a directory if it doesn't already exist
    if not os.path.exists(config["model"]["weights_path"]):
        # may error out due to multiple processes
        try:
            os.mkdir(config["model"]["weights_path"])
        except:
            pass


# load training dataset-------------------------------------------------------

training_data = Data(
    **config["id_dataset"],
    transforms=get_preprocessing_transforms(
        config["id_dataset"]["name"],
        resolution=model.resolution
    )
)
try:
    print(f"Using transforms:\n{training_data.train_set.transforms}")
except:
    # due to Subset from val splitting
    print(
        f"Using transforms:\n{training_data.train_set.dataset.transforms}"
    )

# define pytorch lightning module =============================================

class ClassificationTask(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.top1_calc = TopKError(k=1)
        self.top5_calc = TopKError(k=5)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        err1 = self.top1_calc(targets, outputs)
        err5 = self.top5_calc(targets, outputs)
        self.log_dict(
            {
                "top 1 %error": err1,
                "top 5 %error": err5,
                "loss": loss.detach()
            },
            on_step=True,
            prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):

        last_head = "head"

        inputs, targets = batch
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)


        # measure accuracy and record loss
        # note that outputs should be logits
        # targets should be labels (no distillation)
        err1 = self.top1_calc(targets, outputs)
        err5 = self.top5_calc(targets, outputs)

        self.log_dict(
            {
                f"top 1 validation %error": err1,
                f"top 5 validation %error": err5,
                f"validation loss": loss
            },
            on_epoch=True,
            sync_dist=True,
            prog_bar=True
        )
        return

    def configure_optimizers(self):

        # optimizer and scheduler
        optimizer = OPTIMIZER_MAPPING[config["train_params"]["optimizer"]](
            self.model.parameters(),
            **config["train_params"]["optimizer_params"]
        )

        scheduler = SCHEDULER_MAPPING[config["train_params"]["lr_scheduler"]](
            optimizer,
            **config["train_params"]["lr_scheduler_params"]
        )

        # add a warmup
        if (
            "warmup_epochs" in config["train_params"]
            and
            config["train_params"]["warmup_epochs"] > 0
        ):
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,  # just hardcode this, start small
                total_iters=config["train_params"]["warmup_epochs"]
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[config["train_params"]["warmup_epochs"]]
            )

        return [optimizer], [scheduler]



checkpoint_callback = ModelCheckpoint(
    save_last=True,
    save_on_train_epoch_end=True,
    dirpath=os.path.join(
        config["model"]["weights_path"],
        get_filename(config, seed=config["seed"])
    )+"/"
    # set to be the same as trainer default root dir
)



lr_monitor = LearningRateMonitor(logging_interval='epoch')


if args.amp:
    print("using mixed precision training (amp)")
    precision = 16
else:
    print("using full precision")
    precision = 32


# slurm auto-resubmit job
plugins = None if not args.slurm else [
    SLURMEnvironment(requeue_signal=signal.SIGUSR1)
]

# training loop 
task = ClassificationTask(model)
trainer = pl.Trainer(
    callbacks=[lr_monitor, DeviceStatsMonitor(), checkpoint_callback],
    accelerator=dev.type,
    devices=config["gpu_id"],
    strategy=strategy,
    fast_dev_run=False,
    max_epochs=config["train_params"]["num_epochs"],
    log_every_n_steps=10,
    profiler="simple",
    precision=precision,
    plugins=plugins,
    default_root_dir=os.path.join(
        config["model"]["weights_path"],
        get_filename(config, seed=config["seed"])
    )+"/"
)

# update config, needs this for DDP

try: 
    val_loader = training_data.val_loader
except:
    val_loader = training_data.test_loader
trainer.fit(
    task,
    training_data.train_loader,
    val_loader,
    ckpt_path="last", # default behaviour allows for slurm checkpointing
    # last is a special argument that finds checkpoint from end of last epoch
)
# save at the end for future use
if multi_gpu:
    if trainer.global_rank == 0:
        save_state_dict(model, config=config, is_best=False)
else:
    save_state_dict(model, config=config, is_best=False)
