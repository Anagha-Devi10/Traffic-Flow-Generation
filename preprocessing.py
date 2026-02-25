import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd

file_path_1=glob("/content/Data/402510/*")

def load_data(file_path):
  dataset=[]
  for file in file_path:
    dataset.append(pd.read_csv(file))
  full_data=pd.concat(dataset,axis=0,ignore_index=True)
  return full_data
full_data_1=load_data(file_path_1)

cols_1=list(full_data_1.columns)
data1 = full_data_1[['Flow (Veh/5 Minutes)']].values
data1=np.array(data1).reshape(-1,1)

scaler=MinMaxScaler(feature_range=(-1,1))
Data_scaled_1=scaler.fit_transform(data1)

def create_sequences(data, time_steps, stride=48):
    X= []
    for i in range(0,len(data) - time_steps + 1, stride):
        X.append(data[i:i + time_steps])
    return np.stack(X)

time_steps = 288
X= create_sequences(Data_scaled_1, time_steps, stride=288)
