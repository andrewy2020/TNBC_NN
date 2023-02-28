from pathlib import Path
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler

from parse_mat import parse_mat
from tnbc_nn import *

if __name__ == "__main__":
    # path to matlab workspace
    data_folder = Path("data/")
    filename = "atezolizumab_rngdefault_500"

    filepath = data_folder / (filename + '.mat')

    # # convert .mat to dataframe
    # df = parse_mat(filename + '.mat')

    # convert csv to dataframe
    # to convert .mat to .csv, run parse_mat.py
    df = pd.read_csv(data_folder / (filename + '.csv'))

    # Hyperparameters
    k_folds = 5
    num_epochs = 10
    batch_size = 1
    learning_rate = 1e-4
    loss_fn = nn.CrossEntropyLoss()

    # For fold results
    results = {}

    torch.manual_seed(42)

    # Use GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kfold = KFold(n_splits=k_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(df)):
        # Print fold
        print(f'FOLD {fold+1}')
        print('--------------------------------')

        # separate dataframe into features and labels
        features = df.drop('RECIST', axis=1)
        labels = df['RECIST']

        # convert from dataframes to tensors
        features = torch.tensor(features.values)
        labels = torch.tensor(labels.values)

        dataset = TensorDataset(features, labels)

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_idx)
        test_subsampler = SubsetRandomSampler(test_idx)
        
        # Define data loaders for training and testing data in this fold
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

        # Initialize neural network
        model = NeuralNetwork()

        # Move to chosen device
        # model.to(device)

        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Run the training loop for defined number of epochs
        for epoch in range(num_epochs):

            # Print epoch
            print(f'Starting epoch {epoch+1}')

            
                    
        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')
        
        # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        torch.save(model.state_dict(), save_path)

        # Evaluationfor this fold
        correct, total = 0, 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(test_dataloader, 0):

                # Get inputs
                inputs, targets = data

                # Generate outputs
                outputs = model(inputs.to(torch.float32))

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)
        
    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')


