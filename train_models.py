# --------------------------------------------------------------------------------
# built-in imports
# --------------------------------------------------------------------------------
import os
import sys
import time
import random

# --------------------------------------------------------------------------------
# standard imports
# --------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# working with images
import cv2
import imageio as iio
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# torchvision
import torchvision
import torchvision.transforms as transforms
# torchsummary
import torchsummary
# interactive progress bar
from tqdm import notebook
# debugging
import ipdb

# --------------------------------------------------------------------------------
# custom imports
# --------------------------------------------------------------------------------

# losses
from utils.metrics import (
    iou_pytorch_eval, IoULoss, IoUBCELoss
)
from utils.metrics import (
    iou_pytorch_test, dice_pytorch_test, precision_pytorch_test, recall_pytorch_test, fbeta_pytorch_test, accuracy_pytorch_test
)
# dataset
from utils.dataset import myDataSet
# transforms
from utils.transforms import SIZE, resize_transform, train_transforms, test_transforms
# models
from models.unet import UNet, UNet_attention

# --------------------------------------------------------------------------------
# train and validation loop functions
# --------------------------------------------------------------------------------



# --------------------------------------------------------------------------------
# check settings
# --------------------------------------------------------------------------------

def check_settings(settings):
    # check if settings are correct
    assert isinstance(settings["gpu_index"], int)
    assert settings["gpu_index"] >= 0
    assert isinstance(settings["num_cpu_workers_for_dataloader"], int)
    assert settings["num_cpu_workers_for_dataloader"] > 0
    assert isinstance(settings["batch_size"], int)
    assert settings["batch_size"] > 0

    assert os.path.isdir(settings["images_dir_path"])
    assert os.path.isdir(settings["masks_dir_path"])
    assert os.path.isfile(settings["train_ids_txt"])
    assert os.path.isfile(settings["valid_ids_txt"])

    assert settings["model_architecture"] in ["UNet", "UNet_attention"]
    assert settings["loss_function"] in ["IoULoss", "BCEWithLogitsLoss", "IoUBCELoss"]
    assert settings["training_augmentation"] in [True, False]
    assert settings["model_name"] is not None
    assert settings["model_name"] != ""


# --------------------------------------------------------------------------------
# main function
# --------------------------------------------------------------------------------

def main():

    SETTINGS = {
        "gpu_index": 0, # leave as 0 if you do not have a GPU or only have 1 GPU
        "num_cpu_workers_for_dataloader": 4,
        "batch_size": 8,

        "image_channels": 3,
        "mask_channels": 1,
        "images_dir_path": "data/train-val/images",
        "masks_dir_path": "data/train-val/masks",
        "train_ids_txt": "train-val-split/train.txt",
        "valid_ids_txt": "train-val-split/val.txt",
        
        "model_architecture": "UNet", # "UNet_attention"
        "loss_function": "IoULoss", # IoULoss, BCEWithLogitsLoss, IoUBCELoss
        "training_augmentation": False,
        "model_name": "UNet_IoULoss_baseline", # UNet_BCELoss_baseline, UNet_IoUBCELoss_baseline, UNet_BCEWithLogitsLoss_augmented, UNet_IoULoss_augmented, UNet_IoUBCELoss_augmented, UNet_BCEWithLogitsLoss_attention, UNet_IoULoss_attention, UNet_IoUBCELoss_attention
    }
    
    # set seeds for reproducibility during training
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

    # make the device
    device_type_str = "cuda" if torch.cuda.is_available() else "cpu" # select device for training, i.e. gpu or cpu
    print(device_type_str)
    device_str = f"{device_type_str}:{SETTINGS['gpu_index']}" if device_type_str == "cuda" else device_type_str
    print(device_str)

    # Model Architecture
    if SETTINGS["model_architecture"] == "UNet":
        model = UNet(channel_in=SETTINGS["image_channels"], channel_out=SETTINGS["mask_channels"])
    elif SETTINGS["model_architecture"] == "UNet_attention":
        model = UNet_attention(channel_in=SETTINGS["image_channels"], channel_out=SETTINGS["mask_channels"])
    else:
        raise NotImplementedError
    model = model.to(device_str) # load model to DEVICE
    print(torchsummary.summary(model, (SETTINGS["image_channels"], SIZE[0], SIZE[1]), device=device_type_str))
    
    # Optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-8)

    # Criterion
    if SETTINGS["loss_function"] == "IoULoss":
        criterion = IoULoss()
    elif SETTINGS["loss_function"] == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif SETTINGS["loss_function"] == "IoUBCELoss":
        criterion = IoUBCELoss()
    else:
        raise NotImplementedError


    # Augmentation depends on model name
    if SETTINGS["training_augmentation"]:
        model_train_trasnform = train_transforms 
    else:
        model_train_trasnform = test_transforms


    # pre-defined split - load as list
    with open(SETTINGS["train_ids_txt"], 'r') as f:
        ids_train = [l.strip()+'.jpg' for l in f]
    with open(SETTINGS["valid_ids_txt"], 'r') as f:
        ids_val = [l.strip()+'.jpg' for l in f]
    custom_dataset_train = myDataSet(ids_train, SETTINGS["images_dir_path"], SETTINGS["masks_dir_path"], transforms=model_train_trasnform)
    custom_dataset_valid = myDataSet(ids_val, SETTINGS["images_dir_path"], SETTINGS["masks_dir_path"], transforms=test_transforms)
    print(f"My custom train-dataset has {len(custom_dataset_train)} elements")
    print(f"My custom valid-dataset has {len(custom_dataset_valid)} elements")
    # Create dataloaders from datasets with the native pytorch functions
    dataloader_train = torch.utils.data.DataLoader(
        custom_dataset_train, batch_size=SETTINGS["batch_size"], shuffle=False, num_workers=SETTINGS["num_cpu_workers_for_dataloader"])
    dataloader_valid = torch.utils.data.DataLoader(
        custom_dataset_valid, batch_size=SETTINGS["batch_size"], shuffle=False, num_workers=SETTINGS["num_cpu_workers_for_dataloader"])
    print(f"My custom train-dataloader has {len(dataloader_train)} batches, batch_size={dataloader_train.batch_size}")
    print(f"My custom valid-dataloader has {len(dataloader_valid)} batches, batch_size={dataloader_valid.batch_size}")


if __name__ == "__main__":
    main()