# Checkpoints for models

Best checkpoint file from the rerun in March 2023 (UNet_IoUBCELoss_augmented.pth) can be downloaded from [here](https://drive.google.com/drive/folders/1_TLsaOR-H75X8zHtoztgJSjDOsTNP5Xy?usp=sharing).

It was obtained by running:
```
python train_models.py --loss_function="IoUBCELoss" --training_augmentation=1
```

num_epochs=100, patience=10, model_architecture=UNet,

## Contents of the chekpoint file

When loaded with `torch.load()`, the checkpoint file contains a dictionary with:

- Validation IoU score after binarizing the masks (`iou`)
- Best (maximum Validation IoU) epoch index which ranges from 0 to total_epochs (`epoch`)
- Model weights from the best epoch (`net`)
- training losses (`train_losses`) and validation losses (`val_losses`). They can be used to plot the learning curves without rerunning the training process.
