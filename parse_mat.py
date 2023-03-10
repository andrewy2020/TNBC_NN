import scipy.io as sio
import pandas as pd
import numpy as np
from pathlib import Path

def mat_to_df(filename: str | Path) -> pd.DataFrame:

    # if no squeeze, values are wrapped in [[]] for some reason...
    mat_contents = sio.loadmat(filename, squeeze_me=True)

    # dataframe for input features
    # using item() after squeeze restores proper dimensions; otherwise ()
    features = mat_contents['params_in']['all'].item()
    param_names = mat_contents['params_in']['names'].item()
    df = pd.DataFrame(features, columns=param_names)

    # add labels to dataframe
    labels = mat_contents['params_out']['RECIST'].item()
    df['RECIST'] = labels

    return df

if __name__ == "__main__":
    # path to matlab workspace
    data_folder = Path("data/")

    filenames = ["atezolizumab_rngdefault_2500", "atezolizumab_rng1_2500", "atezolizumab_rng2_2500", "atezolizumab_rng3_2500", "atezolizumab_rng4_2500"]
    frames = []

    for filename in filenames:
        filepath = data_folder / (filename + '.mat')
        # convert .mat to dataframe
        df = mat_to_df(filepath)
        frames.append(df)
    
        df.to_csv(data_folder / (filename + '.csv'), index=False)

    df_tot = pd.concat(frames)
    df_tot.to_csv(data_folder / ("atezolizumab_rngdefault1234_12500" + '.csv'), index=False)