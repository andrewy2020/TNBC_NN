from pathlib import Path
import pandas
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler

from parse_mat import parse_mat
from tnbc_nn import *

#importing Libraries
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Creating the Model
class ANN_model(nn.Module):
    def __init__(self,input_features=8,hidden1=20, hidden2=10,out_features=1):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features,hidden1)
        self.f_connected2 = nn.Linear(hidden1,hidden2)
        self.out = nn.Linear(hidden2,out_features)
        
    def forward(self,x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = F.sigmoid(self.out(x))
        return x


if __name__ == "__main__":
    # path to matlab workspace
    data_folder = Path("data/")
    filename = "PimaDiabetes"

    # filepath = data_folder / (filename + '.mat')

    # # convert .mat to dataframe
    # df = parse_mat(filename + '.mat')

    # convert csv to dataframe
    # to convert .mat to .csv, run parse_mat.py
    df = pd.read_csv(data_folder / (filename + '.csv'))

    # Hyperparameters
    k_folds = 2
    num_epochs = 100
    batch_size = 20
    learning_rate = 1e-2
    loss_fn = nn.BCEWithLogitsLoss()

    # For fold results
    results = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

    torch.manual_seed(42)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    features = df.drop('Outcome', axis=1)
    labels = df.loc[:, ['Outcome']] # extra bracket to keep dataframe 2D

    features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.25,random_state=2) 

    # convert from dataframes to tensors
    features_train = torch.tensor(features_train.values).float()
    labels_train = torch.tensor(labels_train.values).float()

    features_test = torch.tensor(features_test.values).float()
    labels_test = torch.tensor(labels_test.values).float()

    train_dataset = TensorDataset(features_train, labels_train)
    test_dataset = TensorDataset(features_test, labels_test)
    
    # Define data loaders for training and testing data in this fold
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize neural network
    # model = NeuralNetwork().to(device)
    # model = ANN_model().to(device)

    model = nn.Sequential(
        nn.Linear(8, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
        # nn.ReLU(),
        # nn.Linear(10, 1),
        ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Run the training loop for defined number of epochs
    for epoch in range(num_epochs):

        train_loss, train_acc = train_epoch(train_dataloader, model, loss_fn, optimizer, device)
        test_loss, test_acc = test_epoch(test_dataloader, model, loss_fn, device)

        print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                            num_epochs,
                                                                                                            train_loss,
                                                                                                            test_loss,
                                                                                                            train_acc * 100,
                                                                                                            test_acc * 100))
                
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)  