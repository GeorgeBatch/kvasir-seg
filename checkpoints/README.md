# Checkpoints for models

Checkpoint files can be downloaded from [here](https://unioxfordnexus-my.sharepoint.com/:f:/g/personal/mans3968_ox_ac_uk/EgXmR8ZB42xFlNZp9E73alIBjR-C3P_NIS9v7NHg_aTUjw).

## Contents of Each File

When loaded with `torch.load()`, each checkpoint file contains a dictionary with:

- Validation IoU score after binarizing the masks (`iou`)
- Best (maximum Validation IoU) epoch index which ranges from 0 to total_epochs (`epoch`)
- Model weights from the best epoch (`net`)

Some of the checkpoint files also include training (`train_losses`) and validation losses (`val_losses`). They can be used to plot the learning curves without rerunning the training process.

## Directory Contents (after downloading files)

Baseline corresponds to the standard U-Net model run for 100 epochs with early stopping. loss recorded in the file name.

- ckpt_UNet_IoULoss_baseline.pth
- ckpt_UNet_BCELoss_baseline.pth
- ckpt_UNet_IoUBCELoss_baseline.pth

Beseline model with IoU loss performs best on the Validation set, so this loss was taken for further experiments.

Add augmentation:

- ckpt_UNet_IoULoss_augmented.pth

Add a form of "attention" on the upsampling path.

- ckpt_UNet_IoULoss_attention.pth
