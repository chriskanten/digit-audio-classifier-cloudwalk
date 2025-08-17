from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_digit_dataset():

    train_dataset = load_dataset("mteb/free-spoken-digit-dataset", split="train")
    teset_dataset = load_dataset("mteb/free-spoken-digit-dataset", split="test")

    # Convert train_dataset to numpy arrays
    train_features = np.array([item["audio"]["array"] for item in train_dataset])
    train_labels = np.array([item["label"] for item in train_dataset])

    # Convert train_dataset to numpy arrays
    test_features = np.array([item["audio"]["array"] for item in teset_dataset])
    test_labels = np.array([item["label"] for item in teset_dataset])

    # Create datasets
    train_dataset = CustomDataset(train_features, train_labels)
    test_dataset = CustomDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader
