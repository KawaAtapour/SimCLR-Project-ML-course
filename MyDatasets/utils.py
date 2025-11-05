from torchvision import transforms
import torch
from PIL import Image


class SimCLRTransform:
    def __init__(self, image_size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2023, 0.1994, 0.2010])
        ])


    def __call__(self, x):
        return self.transform(x), self.transform(x)



class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform


    def __getitem__(self, index):
        sample = self.base_dataset[index]

        image = sample['image']

        x_i, x_j = self.transform(image)
        return image, x_i, x_j


    def __len__(self):
        return len(self.base_dataset)











