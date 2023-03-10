import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.preprocessing import OneHotEncoder, StandardScaler

class BinaryClassNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(38, 26),
            nn.ReLU(),
            nn.Linear(26, 1),
        )

    def forward(self, x):
        return self.layers(x)

class MultiClassNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(38, 26),
        nn.ReLU(),
        nn.Linear(26, 3),
        )

    def forward(self, x):
        return self.layers(x)

class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx, :]
        label = self.labels[idx]

        return feature, label

def normalize(df: pd.DataFrame, scaler: StandardScaler=None) -> tuple[pd.DataFrame, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
    else:
        df = scaler.transform(df)

    return df, scaler

def remove_nonpatients(df: pd.DataFrame) -> pd.DataFrame:
    
    df['RECIST'].replace('NP', np.nan, inplace=True)

    # remove rows of non-patients
    df.dropna(subset=['RECIST'],inplace = True)

    # reset indices; otherwise drops create gaps
    df.reset_index(drop=True, inplace=True)

    return df

def encode_recist(df: pd.DataFrame, binary: bool = True):

    if (binary):
        # encode RECIST values for binary classification
        df['RECIST'] = df['RECIST'].replace('CR/PR', 1)
        df['RECIST'] = df['RECIST'].replace(['SD', 'PD'], 0)
    else:
        # encode RECIST values for multi-class classification
        # use ones hot encoding to avoid model predicting labels as sums
        enc = OneHotEncoder()
        enc_labels = enc.fit_transform(df.loc[:, ['RECIST']]).toarray()

        enc_labels = pd.DataFrame(enc_labels, columns=enc.get_feature_names_out())

        df = df.drop("RECIST", axis=1)
        df = df.join(enc_labels)
       
    return df

def train_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    total_loss, correct = 0, 0

    model.train()
    for features, labels in dataloader:

        features, labels = features.to(device), labels.to(device)

        # Compute prediction and loss
        pred = model(features)
        loss = loss_fn(pred, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.sigmoid().round() == labels).sum().item()

    avg_loss = total_loss / num_batches
    accuracy = correct / size

    return avg_loss, accuracy


def test_epoch(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    total_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)

            pred = model(features)
            total_loss += loss_fn(pred, labels).item()

            correct += (pred.sigmoid().round() == labels).sum().item()

    avg_loss = total_loss / num_batches
    accuracy = correct / size

    return avg_loss, accuracy