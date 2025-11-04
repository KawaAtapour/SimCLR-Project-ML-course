import numpy as np
import datasets
import torch
from datasets import load_dataset as hf_load_dataset, DatasetDict, Dataset
from collections import defaultdict
import random




######################################################################################################
######################################################################################################
def ddf(x):
    x = datasets.Dataset.from_dict(x)
    x.set_format("torch")
    return x

######################################################################################################
######################################################################################################
def shuffling(a, b):
    return np.random.randint(0, a, b)

######################################################################################################
######################################################################################################
def normalization(batch):
    batch["image"] = batch["image"].float() / 255.0
    return batch

######################################################################################################
######################################################################################################

def prepare_dataset(data, num_classes, samples_per_class):
    # Ensure correct column names
    if "image" not in data.column_names:
        data = data.rename_column(data.column_names[0], "image")
    if "label" not in data.column_names:
        data = data.rename_column(data.column_names[1], "label")

    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(data["label"]):
        class_to_indices[label].append(idx)

    selected_indices = []
    for label in range(num_classes):
        indices = class_to_indices[label]
        selected = random.sample(indices, min(samples_per_class, len(indices)))
        selected_indices.extend(selected)

    sampled_data = data.select(selected_indices)
    sampled_data.set_format("torch", columns=["image", "label"])

    # Normalize if needed
    if sampled_data["image"].max() > 1:
        sampled_data = sampled_data.map(normalization)

    return sampled_data

######################################################################################################
######################################################################################################


def load_dataset(num_train_samples, num_test_samples):
    loaded_dataset = hf_load_dataset("cifar10", split=['train[:100%]', 'test[:100%]'])

    name_classes = loaded_dataset[0].features["label"].names
    num_classes = len(name_classes)

    samples_per_class_train = num_train_samples // num_classes
    samples_per_class_test = num_test_samples // num_classes

    train_data = prepare_dataset(loaded_dataset[0], num_classes, samples_per_class_train)
    test_data = prepare_dataset(loaded_dataset[1], num_classes, samples_per_class_test)

    dataset = DatasetDict({"train": train_data, "test": test_data})

    return dataset, num_classes, name_classes

######################################################################################################
######################################################################################################


