import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)
        
    x = scaler.fit_transform(x)
    data = np.hstack((x,np.reshape(y, (-1,1))))
    return data, x, y

def data_split():
    cols = ["ID","cThick","UCSize", "UCShape", "Adhesion", "CECSize", "Bare", "Bland", "Normal", "Mitoses","class"]
    df = pd.read_csv("breast-cancer-wisconsin.data", names=cols)
    df.drop("Bare", inplace=True, axis=1)
    df.head()

    df["class"] = df["class"].map({2: 0, 4: 1})

    train, valid, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

    return train, valid, test