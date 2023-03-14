
from typing import final
import torch
import torchvision as tv
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.imagenet import ImageNet
import os
import numpy as np
try:
    import dataflow as td
    can_fast = True
except:
    can_fast = False
from io import BytesIO
from PIL import Image
from torchvision.transforms.functional import InterpolationMode





TRAIN_DATASETS = [
"imagenet",
]





# https://github.com/pytorch/vision/issues/39#issuecomment-403701432
# https://paperswithcode.github.io/torchbench/imagenet/
# not necessarily the same as all papers
# use bicubic like efficientnet
# https://github.com/tensorflow/tpu/blob/732902a457b2a8924f885ee832830e1bf6d7c537/models/official/efficientnet/preprocessing.py#L88
CROP_FRACTION = 0.875
def imagenet_transforms(train_res=224, test_res=None):
    test_res = train_res if test_res is None else test_res
    trans = {
        "train": tv.transforms.Compose(
            [
                tv.transforms.RandomResizedCrop(
                    train_res, interpolation=InterpolationMode.BICUBIC
                ),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
            ]
        ),
        "test": tv.transforms.Compose(
            [
                tv.transforms.Resize(
                    round(test_res/CROP_FRACTION), interpolation=InterpolationMode.BICUBIC
                ),
                tv.transforms.CenterCrop(test_res),
                tv.transforms.ToTensor(),
            
            ]
        ),
        "norm":  tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    }
    return trans
# PIL image (0-255) mapped to [0,1], then mean subtracted and std divided 
imagenet_mean=torch.tensor([0.485, 0.456, 0.406])
imagenet_std=torch.tensor([0.229, 0.224, 0.225])


def dataset_name_transform_mapping(train_res=224, test_res=None):
    transform_mapping = {
        "imagenet": imagenet_transforms(train_res=train_res, test_res=test_res),
        "inaturalist": imagenet_transforms(train_res=train_res, test_res=test_res),
        "openimage-o": imagenet_transforms(train_res=train_res, test_res=test_res),
    }
    return transform_mapping



# for tables/plots
DATA_NAME_MAPPING = {
    "imagenet": "ImageNet-1k",
    "inaturalist": "iNaturalist",
    "openimage-o": "Openimage-O"
}




# get transforms for pre-processing
def get_preprocessing_transforms(
    dataset_name, id_dataset_name=None, 
    resolution=224
) -> dict:
    """Get preprocessing transforms for a dataset.
    
    If the dataset is OOD then the ID/training set's normalisation values will
    be used (as if preprocessing is part of network input layer).
    """
    dataset_transforms = dataset_name_transform_mapping(
        train_res=resolution, test_res=resolution # the same for now
    )[dataset_name]

    train_transforms = dataset_transforms["train"]
    # when the dataset itself is the ID dataset
    if id_dataset_name is None:
        transforms = {
            "train": tv.transforms.Compose(
                [
                    train_transforms,
                    dataset_transforms["norm"]
                ]
            ),
            "test": tv.transforms.Compose(
                [
                    dataset_transforms["test"],
                    dataset_transforms["norm"]
                ]
            )
        }
    else:

        # use the in distribution/training sets values for testing ood
        id_dataset_transforms = dataset_name_transform_mapping(
            train_res=resolution
        )[id_dataset_name]
        transforms = {
            "train": tv.transforms.Compose(
                [
                    dataset_transforms["train"],
                    id_dataset_transforms["norm"]
                ]
            ),
            "test": tv.transforms.Compose(
                [
                    dataset_transforms["test"],
                    id_dataset_transforms["norm"]
                ]
            )
        }
    return transforms

# Data object -----------------------------------------------------------------

class Data:
    """Class that contains a datasets + loaders as well as information
    about the dataset, e.g. #samples, transforms for data augmentation.
    Allows the division of the training set into train and validation sets.
    """
    def __init__(
        self,
        name: str, 
        datapath: str,
        download=False,
        batch_size=64,
        test_batch_size=None,
        num_workers=8,
        drop_last=False,
        transforms={"train":None, "test":None},
        target_transforms={"train": None, "test": None},
        val_size=0,
        num_classes=None,
        test_only=False, 
        test_shuffle=False,
        fast=False, # only applies to imagenet loading, DISABLED (does nothing)
        indices=None,
        idx_path=None,
        pin_memory=True,
        **data_kwargs
    ) -> None:
        self.name = name
        self.datapath = datapath
        self.download = download
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size if (
            test_batch_size is not None
        ) else batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last 
        self.transforms = transforms

        # stays around in namespace between objects for some reason
        self.target_transforms = target_transforms.copy() 
        self.val_size = val_size
        self.num_classes = num_classes # this overwrites defaults
        self.test_only = test_only
        self.fast = fast 
        self.test_shuffle = test_shuffle
        self.data_kwargs = data_kwargs
        self.indices = indices
        self.idx_path = idx_path
        self.pin_memory = pin_memory


        # get datasets and dataloaders

        # training/in distribution sets ---------------------------------------
        
        if self.name == "imagenet":

            self.num_classes = 1000 if (
                self.num_classes is None
            ) else self.num_classes

            
            try:
                self.test_set = tv.datasets.ImageNet(
                    root=self.datapath,
                    split="val",
                    transform=self.transforms["test"],
                    target_transform=self.target_transforms["test"]
                )

            # unzipped folders
            except:
                try:
                    self.test_set = tv.datasets.ImageFolder(
                        root=os.path.join(self.datapath, "validation"),
                        transform=self.transforms["test"],
                        target_transform=self.target_transforms["test"]
                    )
                # different folder name
                except:
                    self.test_set = tv.datasets.ImageFolder(
                        root=os.path.join(self.datapath, "val"),
                        transform=self.transforms["test"],
                        target_transform=self.target_transforms["test"]
                    )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=self.test_shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last, pin_memory=self.pin_memory,
            )

            # train
            if not self.test_only:
                try:
                    self.train_set = tv.datasets.ImageNet(
                        root=self.datapath,
                        split="train",
                        transform=self.transforms["train"],
                        target_transform=self.target_transforms["train"]
                    )
                except:
                    self.train_set = tv.datasets.ImageFolder(
                        root=os.path.join(self.datapath, "train"),
                        transform=self.transforms["train"],
                        target_transform=self.target_transforms["train"]
                    )
                print(f"{len(self.train_set)} samples in imagenet")
                indices = torch.randperm(len(self.train_set))
                if self.idx_path is None:
                    self.idx_path = self.datapath
                idx_path = os.path.join(self.idx_path, 'index.pth')
                if os.path.exists(idx_path):
                    print('!!!!!! Load train_set_index !!!!!!')
                    indices = torch.load(idx_path)
                else:
                    print('!!!!!! Save train_set_index !!!!!!')
                    torch.save(indices, idx_path)


                # allows for external override 
                # msdnet indices are saved in a weird place for pretrained 
                # models
                # using the same val split method in the MSDNet repo
                # https://github.com/kalviny/MSDNet-PyTorch/blob/master/dataloader.py
                if self.indices is None:
                    self.indices = indices
            
                if val_size > 0:
                    assert (
                        val_size <= len(self.train_set)
                    ), "val size larger than training set"

                    # train/val split
                    self.train_indices = self.indices[0:-val_size]
                    self.val_indices = self.indices[-val_size:]
                    
                    # train set with test transforms
                    try:
                        self.val_set = tv.datasets.ImageNet(
                            root=self.datapath,
                            split="train",
                            transform=self.transforms["test"],
                            target_transform=self.target_transforms["test"]
                        )
                    except:
                        self.val_set = tv.datasets.ImageFolder(
                            root=os.path.join(self.datapath, "train"),
                            transform=self.transforms["test"],
                            target_transform=self.target_transforms["test"]
                        )
                    
                    self.val_set = Subset(self.val_set, self.val_indices)
                    self.val_set.targets = [
                        self.train_set.targets[idx] for idx in self.val_indices
                    ]

                    self.val_loader = DataLoader(
                        self.val_set,
                        batch_size=self.test_batch_size,
                        num_workers=self.num_workers,
                        drop_last=self.drop_last, pin_memory=self.pin_memory,
                        shuffle=False,
                    )

                else:
                    self.train_indices = self.indices

                self.train_set = Subset(self.train_set, self.train_indices)
                self.train_loader = DataLoader(
                    self.train_set,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=self.drop_last, pin_memory=self.pin_memory,
                    shuffle=True,
                )

        
        # OOD detection for imagenet-------------------------------------------
       

        if self.name in [
            "inaturalist"
        ]:

            # these ones are 10,000 in size
            self.test_only = True
            print(f"{self.name} is only available for testing/OOD detection")

            self.test_set = tv.datasets.ImageFolder(
                root=self.datapath,
                transform=self.transforms["test"],
                target_transform=self.target_transforms["test"]
            )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=self.test_shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last, pin_memory=self.pin_memory,
            )


        if self.name == "openimage-o":
            self.test_only = True
            print(f"{self.name} is only available for testing/OOD detection")

            # this only gets a subset of the images in the directory
            self.test_set = ImageFilelist(
                root=os.path.join(self.datapath, "test"),
                flist=os.path.join(self.datapath, "openimages_datalist.txt"),
                transform=self.transforms["test"],
                target_transform=self.target_transforms["test"]
            )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=self.test_shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last, pin_memory=self.pin_memory,
            )



# for openimage-o (loads from subset of images)


def default_loader(path):
	return Image.open(path).convert('RGB')
def default_flist_reader(flist):
	"""
	flist format: impath label\nimpath label\n
	"""
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			data = line.strip().rsplit(maxsplit=1)
			if len(data) == 2:
				impath, imlabel = data
			else:
				impath, imlabel = data[0], 0
			imlist.append( (impath, int(imlabel)) )

	return imlist

class ImageFilelist(torch.utils.data.Dataset):
	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=default_loader):
		self.root   = root
		self.imlist = flist_reader(flist)
		self.transforms = transform
		self.target_transforms = target_transform
		self.loader = loader

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(os.path.join(self.root,impath))
		if self.transforms is not None:
			img = self.transforms(img)
		if self.target_transforms is not None:
			target = self.target_transforms(target)

		return img, target

	def __len__(self):
		return len(self.imlist)

class ImageFolderSubset(ImageFolder):
    """Only recognises a subset of classes
    Uses class names (directory names) rather than indices
    """
    def __init__(
        self, 
        root: str, 
        class_subset = None,
        transform = None, 
        target_transform = None, 
        is_valid_file = None,
        target_class_to_idx = None
    ):
        super().__init__(root, transform, target_transform, is_valid_file=is_valid_file)
        if class_subset is not None:

            
            # indices in loaded dataset 
            subset_ids = [self.class_to_idx[cls] for cls in class_subset]

            # index to index, loaded to target dataset
            if target_class_to_idx is None:
                class_mapping = {
                    self.class_to_idx[cls]: i 
                    for i, cls in enumerate(class_subset)
                }
            else:
                class_mapping = {
                    self.class_to_idx[cls]: target_class_to_idx[cls]
                    for i, cls in enumerate(class_subset)
                }
            
            # map loaded labels to target labels
            self.samples = [
                (s[0], class_mapping[s[1]]) 
                for s in self.samples 
                if s[1] in subset_ids
            ]
            self.imgs = self.samples
            self.targets = [s[1] for s in self.samples]

