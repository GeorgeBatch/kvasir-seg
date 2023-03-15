import torchvision.transforms as transforms

SIZE = (256, 256)
INTERPOLATION_MODE = transforms.InterpolationMode.NEAREST

resize_transform = transforms.Resize(SIZE, interpolation=INTERPOLATION_MODE)

# TODO: check if the padding argument changed since 2020
train_transforms = transforms.Compose([
                           transforms.Resize(SIZE, interpolation=INTERPOLATION_MODE),
                           transforms.RandomRotation(180),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(SIZE, padding = 10), # needed after rotation (with original size)
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize(SIZE, interpolation=INTERPOLATION_MODE),
                       ])


if __name__ == "__main__":
    for i, mode in enumerate(transforms.InterpolationMode):
        print(i, mode)
    print("\nPreviously, `interpolation=0` was used. Now it was substituted by transforms.InterpolationMode.NEAREST - the one with index 0.")