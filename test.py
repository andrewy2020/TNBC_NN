from pathlib import Path
from numpy import fabs
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

    df = pd.read_csv(data_folder / (filename + '.csv'))

    df = encode_recist(df, binary=False)

    print(df)