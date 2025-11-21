from torchvision import transforms
import torch
from PIL import Image




class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        image = self.base_dataset[index]["image"]  
        x_i, x_j = self.transform(image)
        return image, x_i, x_j

    def __len__(self):
        return len(self.base_dataset)




class SimCLRTransform:
    def __init__(self, image_size, dataset_name):
        if "imagenet" in dataset_name:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif "cifar" in dataset_name:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        # ✅ Convert tensor to PIL and force RGB
        if isinstance(x, torch.Tensor):
            x = transforms.ToPILImage()(x)
        x = x.convert("RGB")  # ✅ Always 3 channels
        return self.transform(x), self.transform(x)



