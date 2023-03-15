import torchvision.transforms as transforms

SIZE = 256

resize_transform = transforms.Resize(SIZE, interpolation=transforms.InterpolationMode.NEAREST)


train_transforms = transforms.Compose([
                           transforms.Resize(SIZE, interpolation=transforms.InterpolationMode.NEAREST),
                           transforms.RandomRotation(180),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(SIZE, padding = 10), # needed after rotation (with original size)
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize(SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
                       ])


if __name__ == "__main__":
    for i, mode in enumerate(transforms.InterpolationMode):
        print(i, mode)
    print("\nPreviously, `interpolation=0` was used. Now it was substituted by transforms.InterpolationMode.NEAREST - the one with index 0.")