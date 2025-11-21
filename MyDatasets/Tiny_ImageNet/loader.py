
from datasets import DatasetDict
from datasets import load_dataset as hf_load_dataset
import torch
import random
from collections import defaultdict

######################################################################################################
def normalization(batch):
    batch["image"] = batch["image"].float() / 255.0
    return batch

######################################################################################################
def prepare_dataset(data, num_classes, samples_per_class):
    if "image" not in data.column_names:
        data = data.rename_column(data.column_names[0], "image")
    if "label" not in data.column_names:
        data = data.rename_column(data.column_names[1], "label")

    # ✅ Filter out grayscale images using PIL mode
    def filter_rgb(example):
        # PIL images have .mode attribute; RGB means 3 channels
        return example["image"].mode == "RGB"

    data = data.filter(filter_rgb)

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

    # Normalize
    sampled_data = sampled_data.map(normalization)

    return sampled_data

######################################################################################################
def load_dataset(num_train_samples, num_test_samples):
    dataset_dict = hf_load_dataset("Maysee/tiny-imagenet")

    train_data = dataset_dict["train"]
    test_data = dataset_dict["valid"]

    name_classes = train_data.features["label"].names
    num_classes = len(name_classes)

    samples_per_class_train = num_train_samples // num_classes
    samples_per_class_test = num_test_samples // num_classes

    train_data = prepare_dataset(train_data, num_classes, samples_per_class_train)
    test_data = prepare_dataset(test_data, num_classes, samples_per_class_test)

    return DatasetDict({"train": train_data, "test": test_data}), num_classes, name_classes
