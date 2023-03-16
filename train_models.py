# --------------------------------------------------------------------------------
# built-in imports
# --------------------------------------------------------------------------------
import os
import sys
import copy
import time
import random
import argparse

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
# torchmetrics
import torchmetrics
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
    iou_pytorch_eval, IoULoss, IoUBCELoss, BCEWithLogitsLoss
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

def train_eval_one_epoch(model, optimizer, criterion, dataloader, epoch, device, settings, train_mode):
    if train_mode == True:
        model.train()
    else:
        model.eval()


    total_loss = 0
    total_iou = 0
    for i, (imgs, masks) in enumerate(dataloader):
        batch_size = imgs.shape[0]
        
        imgs, masks = imgs.to(device), masks.to(device) # (batch_size, 3, 256, 256), (batch_size, 1, 256, 256)
        if train_mode:
            prediction = model(imgs)
        else:
            with torch.no_grad():
                prediction = model(imgs)
        # print(prediction.shape) # (batch_size, 1, 256, 256)

        if train_mode:
            optimizer.zero_grad()
            loss = criterion(prediction, masks)
            loss.backward()
            optimizer.step()
        else:
            loss = criterion(prediction, masks)

        batch_loss = loss.item()
        total_loss += batch_loss

        batch_iou = iou_pytorch_eval(prediction, masks, reduction="sum")
        total_iou += batch_iou

        print(f"\r Epoch: {epoch} of {settings['num_epochs']-1}, Iter.: {i+1} of {len(dataloader)}, Avg Batch Loss: {batch_loss / batch_size:.6f}", end="")
        print(f"\r Epoch: {epoch} of {settings['num_epochs']-1}, Iter.: {i+1} of {len(dataloader)}, Avg Batch IoU : {batch_iou  / batch_size:.6f}", end="")

    print()
    avg_loss = total_loss / len(dataloader.dataset)
    avg_iou = total_iou / len(dataloader.dataset)

    prefix = "Train" if train_mode else "Valid"
    print(f"\r Epoch: {epoch} of {settings['num_epochs']-1}, {prefix} Avg Epoch Loss: {avg_loss:.2f}", end="")
    print(f"\r Epoch: {epoch} of {settings['num_epochs']-1}, {prefix} Avg Epoch IoU : {avg_iou:.2f}", end="\n")
    
    return avg_loss, avg_iou
 


# --------------------------------------------------------------------------------
# check settings
# --------------------------------------------------------------------------------

def check_settings(original_settings):
    settings = copy.deepcopy(original_settings)

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
    assert isinstance(settings["training_augmentation"], bool)
    assert settings["model_name"] is not None
    assert settings["model_name"] != ""


# --------------------------------------------------------------------------------
# main function
# --------------------------------------------------------------------------------

# tmux sessions 15, 16, 17, 18 on compg015
# python train_models.py --gpu_index=0 --loss_function="IoULoss" --training_augmentation=0
# python train_models.py --gpu_index=1 --loss_function="BCEWithLogitsLoss" --training_augmentation=0
# python train_models.py --gpu_index=2 --loss_function="IoUBCELoss" --training_augmentation=0
# python train_models.py --gpu_index=3 --loss_function="IoULoss" --training_augmentation=1

def main():
    parser = argparse.ArgumentParser(description='Train and validate a segmentation model on Kvasir-Seg dataset')
    parser.add_argument('--gpu_index', default=0, type=int, help='GPU ID [0|1|2|3]')
    parser.add_argument('--num_cpu_workers_for_dataloader', default=4, type=int, help='Number of CPU workers for dataloader [0|1|2|3|4]')
    parser.add_argument('--batch_size', default=20, type=int, help='Batch size')
    parser.add_argument('--model_architecture', default='UNet', type=str, help='Model architecture [UNet|UNet_attention]')
    parser.add_argument('--loss_function', default='IoULoss', type=str, help='Loss function [IoULoss|BCEWithLogitsLoss|IoUBCELoss]')
    parser.add_argument('--training_augmentation', default=0, type=int, help='Whether to use training augmentation [1|0]')
    
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs [5|100]')
    parser.add_argument('--patience', default=10, type=int, help='Number of patience training epochs [2|10]')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    args = parser.parse_args()

    SETTINGS = {
        "gpu_index": args.gpu_index,
        "num_cpu_workers_for_dataloader": args.num_cpu_workers_for_dataloader,
        "batch_size": args.batch_size,

        "model_architecture": args.model_architecture, # UNet, UNet_attention
        "loss_function": args.loss_function, # IoULoss, BCEWithLogitsLoss, IoUBCELoss
        "training_augmentation": bool(args.training_augmentation), # True, False
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "patience": args.patience,

        "image_channels": 3,
        "mask_channels": 1,
        "images_dir_path": "data/train-val/images",
        "masks_dir_path": "data/train-val/masks",
        "train_ids_txt": "train-val-split/train.txt",
        "valid_ids_txt": "train-val-split/val.txt",
    }
    postfix = "augmented" if SETTINGS["training_augmentation"] else "baseline"
    SETTINGS["model_name"] = f"{SETTINGS['model_architecture']}_{SETTINGS['loss_function']}_{postfix}"
    check_settings(SETTINGS)
    
    # set seeds for reproducibility during training
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

    # make the device
    device_type_str = "cuda" if torch.cuda.is_available() else "cpu" # select device for training, i.e. gpu or cpu
    print("device_type_str:", device_type_str)
    device_str = f"{device_type_str}:{SETTINGS['gpu_index']}" if device_type_str == "cuda" else device_type_str
    print("     device_str:", device_str)

    # Model Architecture
    if SETTINGS["model_architecture"] == "UNet":
        model = UNet(channel_in=SETTINGS["image_channels"], channel_out=SETTINGS["mask_channels"])
    elif SETTINGS["model_architecture"] == "UNet_attention":
        model = UNet_attention(channel_in=SETTINGS["image_channels"], channel_out=SETTINGS["mask_channels"])
    else:
        raise NotImplementedError
    model = model.to(device_str) # load model to DEVICE
    #print(torchsummary.summary(model, (SETTINGS["image_channels"], SIZE[0], SIZE[1]), device=device_type_str))
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay = SETTINGS["weight_decay"])

    # Criterion
    if SETTINGS["loss_function"] == "IoULoss":
        criterion = IoULoss(reduction="sum")
    elif SETTINGS["loss_function"] == "BCEWithLogitsLoss":
        criterion = BCEWithLogitsLoss(reduction="sum")
    elif SETTINGS["loss_function"] == "IoUBCELoss":
        criterion = IoUBCELoss(reduction="sum")
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
        custom_dataset_train, batch_size=SETTINGS["batch_size"], num_workers=SETTINGS["num_cpu_workers_for_dataloader"],
        shuffle=False, drop_last=False)
    dataloader_valid = torch.utils.data.DataLoader(
        custom_dataset_valid, batch_size=SETTINGS["batch_size"], num_workers=SETTINGS["num_cpu_workers_for_dataloader"],
        shuffle=False, drop_last=False)
    print(f"My custom train-dataloader has {len(dataloader_train)} batches, batch_size={dataloader_train.batch_size}")
    print(f"My custom valid-dataloader has {len(dataloader_valid)} batches, batch_size={dataloader_valid.batch_size}")

    # train and evaluate
    train_losses = []
    valid_losses = []
    best_iou = 0
    best_loss = np.Inf
    best_epoch = -1
    state = {}

    for epoch in range(SETTINGS["num_epochs"]):
        epoch_avg_train_loss, epoch_avg_train_iou = train_eval_one_epoch(
            model, optimizer, criterion, dataloader_train, epoch, device=device_str, settings=SETTINGS, train_mode=True)
        epoch_avg_valid_loss, epoch_avg_valid_iou = train_eval_one_epoch(
            model, optimizer, criterion, dataloader_valid, epoch, device=device_str, settings=SETTINGS, train_mode=False)

        train_losses.append(epoch_avg_train_loss)
        valid_losses.append(epoch_avg_valid_loss)

        # save if best results or break is has not improved for {patience} number of epochs
        best_iou = max(best_iou, epoch_avg_valid_iou)
        best_loss = min(best_loss, epoch_avg_valid_loss)
        best_epoch = epoch if best_iou == epoch_avg_valid_iou else best_epoch
        
        # record losses
        state['train_losses'] = train_losses
        state['valid_losses'] = valid_losses
        
        if best_epoch == epoch:
            # print('Saving..')
            state['net'] = model.state_dict()
            state['iou'] = best_iou
            state['epoch'] = epoch
                
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(state, f'checkpoints/{SETTINGS["model_name"]}.pth')
        
        elif best_epoch + SETTINGS['patience'] < epoch:
            print(f"\nEarly stopping. Target criteria has not improved for {SETTINGS['patience']} epochs.\n")
            break

    # load once more and write all the losses down (othw can miss the last 10)
    state = torch.load(f'checkpoints/{SETTINGS["model_name"]}.pth')
    state['train_losses'] = train_losses
    state['val_losses'] = valid_losses
    torch.save(state, f'checkpoints/{SETTINGS["model_name"]}.pth')
    print(f'Best epoch: {best_epoch}, Best IoU: {best_iou}')
    
    # Checks
    model.load_state_dict(torch.load(f'checkpoints/{SETTINGS["model_name"]}.pth')['net'])
    print('Best epoch:', torch.load(f'checkpoints/{SETTINGS["model_name"]}.pth')['epoch'])
    print(f'Validation IoU ({SIZE[0]}x{SIZE[1]}):', torch.load(f'checkpoints/{SETTINGS["model_name"]}.pth')['iou'].item())


if __name__ == "__main__":
    main()

