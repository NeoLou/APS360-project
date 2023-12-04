from common import model
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

class ANN(model.Model):
    def __init__(self, name, batch_size, lr, num_epochs, num_layers, hidden_size, activation):
        super(ANN, self).__init__(name=name,
                                    batch_size=batch_size,
                                    lr=lr,
                                    num_epochs=num_epochs)

        assert num_layers >= 2, "You must initialize with 2 or more layers"
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()

        # add all the hidden layers we want
        # input layer
        self.layers.append(nn.Linear(20, hidden_size))

        # hidden layers
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # output layer (single node)
        self.layers.append(nn.Linear(hidden_size, 1))

        # activation
        self.activ = activation.__name__
        self._activation = activation

    def forward(self, input):
        pass_forward = input # this is what one layer will pass to the next

        for i, layer in enumerate(self.layers):
            pass_forward = layer(pass_forward)
            pass_forward = self._activation(pass_forward)

        return pass_forward

class animeDataset(Dataset):
    def __init__(self, df):
        self.df = self.normalize_cols(df)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, id):
        row = self.df.iloc[id]
        label = torch.tensor(row['popularity']).float()
        data = torch.tensor(row[2:]).float()
        # return data.to('cuda'), label.to('cuda')
        return data, label


    def normalize_cols(self, df):
        for column in df.columns:
            if df[column].max() != 0:
                df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        return df

def get_data_loaders(path, batch_size):
    df = pd.read_csv(path)
    ds = animeDataset(df)

    if 'test' in path:
        return DataLoader(ds, batch_size=batch_size)

    train_size = int(0.8*len(ds))
    val_size = len(ds) - train_size
    train, val = random_split(ds, [train_size, val_size])
    return DataLoader(train, batch_size=batch_size), DataLoader(val, batch_size=batch_size)

