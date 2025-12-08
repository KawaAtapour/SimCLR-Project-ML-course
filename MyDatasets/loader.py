
import torch
from torchvision.datasets import STL10
from torchvision import transforms
from datasets import Dataset, DatasetDict
from collections import defaultdict
import random

# Transform: convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()  # Converts to [0,1] automatically
])

def prepare_dataset_from_torchvision(torch_dataset, num_classes, samples_per_class):
    # Extract images and labels
    images = []
    labels = []
    for img, label in torch_dataset:
        images.append(img)
        labels.append(label)

    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)

    selected_indices = []
    for label in range(num_classes):
        indices = class_to_indices[label]
        selected = random.sample(indices, min(samples_per_class, len(indices)))
        selected_indices.extend(selected)

    sampled_images = [images[i] for i in selected_indices]
    sampled_labels = [labels[i] for i in selected_indices]

    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_dict({"image": sampled_images, "label": sampled_labels})
    hf_dataset.set_format("torch", columns=["image", "label"])
    return hf_dataset

def load_dataset(num_train_samples, num_test_samples):
    # Load STL10 using torchvision
    train_dataset = STL10(root="./data", split="train", download=True, transform=transform)
    test_dataset = STL10(root="./data", split="test", download=True, transform=transform)

    num_classes = 10
    name_classes = [str(i) for i in range(num_classes)]

    samples_per_class_train = num_train_samples // num_classes
    samples_per_class_test = num_test_samples // num_classes

    train_data = prepare_dataset_from_torchvision(train_dataset, num_classes, samples_per_class_train)
    test_data = prepare_dataset_from_torchvision(test_dataset, num_classes, samples_per_class_test)

    dataset = DatasetDict({"train": train_data, "test": test_data})

    return dataset, num_classes, name_classes
