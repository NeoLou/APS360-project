import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

print(SCRIPT_DIR)

from common import model
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe

import numpy as np
import math

glove = GloVe(name='840B', dim=300)

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TransformerEncoder, self).__init__()
        self.linear_q = nn.Linear(input_size, hidden_size).cuda()
        self.linear_k = nn.Linear(input_size, hidden_size).cuda()
        self.linear_v = nn.Linear(input_size, hidden_size).cuda()
        self.linear_x = nn.Linear(input_size, hidden_size).cuda()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True).cuda()
        self.fc = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size)).cuda()
        self.norm = nn.LayerNorm(hidden_size).cuda()
        
    def forward(self, x):
        q, k, v = self.linear_q(x).cuda(), self.linear_k(x).cuda(), self.linear_v(x).cuda()
        attn_output, attn_output_weights = self.attention(q, k, v)
        x = self.norm(self.linear_x(x).cuda() + attn_output.cuda()).cuda()
        x = self.norm(x.cuda() + self.fc(x).cuda()).cuda()
        return x.cuda()

class SynopsisTransformer(model.Model):
    def __init__(self, name, batch_size, lr, num_epochs, input_size, hidden_size, num_class):
        super(SynopsisTransformer, self).__init__(name=name,
                                      batch_size=batch_size,
                                      lr=lr,
                                      num_epochs=num_epochs)
        # layers
        self.encoder = TransformerEncoder(input_size, hidden_size).cuda()
        self.fc = nn.Linear(hidden_size, num_class).cuda()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_class = num_class

    def forward(self, x):
        x = self.encoder(x).cuda()
        return self.fc(x).cuda()
        
class animeDataset(Dataset):
    def __init__(self, df):
        self.df, self.max_len = self.normalize_cols(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        row = self.df.iloc[id]
        label = torch.tensor(row['popularity']).float().cuda()
        data = row['synopsis'].cuda()
        return data, label
    
    def gen_pe(self, max_length, d_model, n):
        # generate an empty matrix for the positional encodings (pe)
        pe = np.zeros(max_length*d_model).reshape(max_length, d_model) 
        # for each position
        for k in np.arange(max_length):
            # for each dimension
            for i in np.arange(d_model//2):
            # calculate the internal value for sin and cos
                theta = k / (n ** ((2*i)/d_model))       
                # even dims: sin   
                pe[k, 2*i] = math.sin(theta) 
                # odd dims: cos               
                pe[k, 2*i+1] = math.cos(theta)
        return pe

    def normalize_cols(self, df):
        cols = ['popularity', 'synopsis']
        df = df[cols]
        df['popularity'] = (df['popularity'] - df['popularity'].min()) / (df['popularity'].max() - df['popularity'].min())
        
        max_len = -1
        tokenizer = get_tokenizer("basic_english")
        for synopsis in df['synopsis']:
            syn_len = len(tokenizer(synopsis))
            if syn_len > max_len:
                max_len = syn_len
                
        n = 10000
        d_model = 300
        encodings = self.gen_pe(max_len, d_model, n)
        encodings = torch.from_numpy(np.transpose(encodings)).double()
        
        for i in range(len(df['synopsis'])):
            df['synopsis'][i] = tokenizer(df['synopsis'][i])
            df['synopsis'][i]
            df['synopsis'][i] += ['<pad>'] * (max_len - len(df['synopsis'][i]))
            df['synopsis'][i] = glove.get_vecs_by_tokens(df['synopsis'][i])
            df['synopsis'][i] = torch.transpose(df['synopsis'][i], 0, 1)
            
            df['synopsis'][i] += encodings
            
            
        return df, max_len

def get_data_loaders(path_to_csv, batch_size=32):
    df = pd.read_csv(path_to_csv)
    ds = animeDataset(df)
    max_len = ds.max_len = ds.max_len
    train_size = int(0.8*len(ds))
    val_size = len(ds) - train_size
    train, val = random_split(ds, [train_size, val_size])
    return DataLoader(train, batch_size), DataLoader(val, batch_size), max_len

