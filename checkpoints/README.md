# Checkpoints for models

Each checkpoint file contains:

- Validation IoU score (after binarizing the masks)
- Best (maximum Validation IoU) epoch index (from 0 to total_epochs)
- Model weights from the best epoch

## Directory Contents

Baseline corresponds to the standard U-Net model run for 50 epochs with loss recorded in the file name.

- ckpt_UNet_IoULoss_baseline.pth
- ckpt_UNet_BCEWithLogitsLoss_baseline.pth
- ckpt_UNet_IoUBCELoss_baseline.pth

Beseline model with IoU loss performes best on the Validation set, so this loss was taken for further experimants.

Add augmentation: 

- ckpt_UNet_IoULoss_augmented.pth

Add a form of "attention" on the upsampling path.

- ckpt_UNet_IoULoss_attention.pth
