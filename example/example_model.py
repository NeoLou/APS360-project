from common import model
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

class Example(model.Model):
    def __init__(self, name, batch_size, lr, num_epochs):
        super(Example, self).__init__(name=name,
                                      batch_size=batch_size,
                                      lr=lr,
                                      num_epochs=num_epochs)

        # layers
        self.fc = nn.Linear(4, 1)

    def forward(self, input):
        return F.relu(self.fc(input))

class animeDataset(Dataset):
    def __init__(self, df):
        self.df = self.normalize_cols(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        row = self.df.iloc[id]
        label = torch.tensor(row['popularity']).float()
        data = torch.tensor(row[1:]).float()
        return data, label

    def normalize_cols(self, df):
        cols = ['popularity', "genres_0_id", "num_episodes", "average_episode_duration", "studios_0_id"]
        df = df[cols]
        for column in cols:
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        return df

def get_data_loaders(path_to_csv):
    df = pd.read_csv(path_to_csv)
    ds = animeDataset(df)
    train_size = int(0.8*len(ds))
    val_size = len(ds) - train_size
    train, val = random_split(ds, [train_size, val_size])
    return DataLoader(train, batch_size=32), DataLoader(val, batch_size=32)

