from os import remove
from pathlib import Path
import pandas
from torch.utils.data import Dataset, DataLoader,TensorDataset,Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from parse_mat import mat_to_df
from tnbc_nn import *

if __name__ == "__main__":
    # path to matlab workspace
    data_folder = Path("data/")
    filename = "atezolizumab_rngdefault1234_12500"

    # filepath = data_folder / (filename + '.mat')

    # # convert .mat to dataframe
    # df = parse_mat(filename + '.mat')

    # convert csv to dataframe
    # to convert .mat to .csv, run parse_mat.py
    df = pd.read_csv(data_folder / (filename + '.csv'))

    # Hyperparameters
    k_folds = 10
    num_epochs = 100
    batch_size = 50
    learning_rate = 1e-4
    loss_fn = nn.BCEWithLogitsLoss()

    # For fold results
    results = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

    torch.manual_seed(42)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kfold = KFold(n_splits=k_folds, shuffle=True)

    # remove NP rows before kfold split
    df = remove_nonpatients(df)

    # K-fold Cross Validation model evaluation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(df)):
        # Print fold
        print(f'FOLD {fold+1}')
        print('--------------------------------')


        

        # separate dataframe into features and labels
        features = df.drop('RECIST', axis=1)
        labels = df.loc[:, ['RECIST']] # extra bracket to keep 2D dataframe instead of 1D series

        # binary or multiclass encoding
        labels = encode_recist(labels)

        # convert from dataframes to tensors
        features = torch.FloatTensor(features.values)
        labels = torch.FloatTensor(labels.values)

        dataset = TensorDataset(features, labels)

        # Low accuracy with SubsetRandomSampler method; not sure why
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        # Define data loaders for training and testing data in this fold
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize neural network
        # model = NeuralNetwork().to(device)
        model = nn.Sequential(
                    nn.Linear(38, 26),
                    nn.ReLU(),
                    nn.Linear(26, 1),
                    # nn.Sigmoid(),
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
        
    # Print fold results
    avg_train_loss = np.mean(results['train_loss'])
    avg_test_loss = np.mean(results['test_loss'])
    avg_train_acc = np.mean(results['train_acc'])
    avg_test_acc = np.mean(results['test_acc'])

    print('Performance of {} fold cross validation'.format(k_folds))
    print("Average Training Loss: {:.4f} \t Average Test Loss: {:.4f} \t Average Training Acc: {:.3f} \t Average Test Acc: {:.3f}".format(avg_train_loss,avg_test_loss,avg_train_acc,avg_test_acc)) 


