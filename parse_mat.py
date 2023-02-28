import scipy.io as sio
import pandas as pd
import numpy as np
from pathlib import Path

def parse_mat(filepath):

    # if no squeeze, values are wrapped in [[]] for some reason...
    mat_contents = sio.loadmat(filepath, squeeze_me=True)

    # dataframe for input features
    # using item() after squeeze restores proper dimensions; otherwise ()
    features = mat_contents['params_in']['all'].item()
    param_names = mat_contents['params_in']['names'].item()
    df = pd.DataFrame(features, columns=param_names)

    # add labels to dataframe
    labels = mat_contents['params_out']['RECIST'].item()
    df['RECIST'] = labels

    # encode RECIST values for binary classification
    df['RECIST'].replace('CR/PR', 1, inplace=True)
    df['RECIST'].replace(['SD', 'PD'], 0, inplace=True)
    df['RECIST'].replace('NP', np.nan, inplace=True)

    # remove rows of non-patients
    df.dropna(subset=['RECIST'],inplace = True)

    # reset indices; otherwise drops create gaps
    df.reset_index(drop=True, inplace=True)

    return df

if __name__ == "__main__":
    # path to matlab workspace
    data_folder = Path("data/")

    filename = "atezolizumab_rngdefault_500"

    filepath = data_folder / (filename + '.mat')

    # convert .mat to dataframe
    df = parse_mat(filepath)
    
    df.to_csv(data_folder / (filename + '.csv'), index=False)




