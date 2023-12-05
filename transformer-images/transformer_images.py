from common import model
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import glob
import pickle

class LinearProjector(nn.Module):
    def __init__(self, N, hidden_size, batch_size=64, input_size=464):
        super(LinearProjector, self).__init__()
        self.N = N # number of patches to split image into
        self.hidden_size = hidden_size
        self._input_size = input_size
        self._P = self._get_P()
        self.batch_size=batch_size
        self._class_embed = self.get_class_embed()

        # layers
        # map image input after patching to D with learnable embeddings
        # need a variable amount if we want to use N as a hyperparam
        self._fc_list = nn.ModuleList([
            nn.Linear((self._P**2)*3, self.hidden_size) for _ in range(N+1)
            ])

    def get_class_embed(self):

        class_embed = (
            torch.randn(self.hidden_size+1)
            .unsqueeze(0)
        )
        class_embed[:, 0] = 0.0
        return nn.Parameter(class_embed).to('cuda')

    def break_into_patches(self, input):
        """break tensor input image into N patches"""
        H, W = tuple(input.shape[-2:])
        assert H == self._input_size, "please make sure inputs are in the right format (batch_size, # channels, H, W) and that img is a square"
        assert W == self._input_size, "please make sure inputs are in the right format (batch_size, # channels, H, W) and that img is a square"
        P = self._P
        assert H%P == 0, "Choose N that image can be evenly split into pls"

        split1 = torch.split(input, P, -1)
        split2 = []
        for split in split1:
            split2.extend(list(torch.split(split, P, -2)))

        if self.training:
            to_return = torch.stack(split2).permute(1, 0, 2, 3, 4)
        else:
            to_return = torch.stack(split2)
        return to_return.to('cuda')

    def _get_P(self):
        return int(np.sqrt((self._input_size**2)/self.N))

    def forward(self, input):
        patches = self.break_into_patches(input)
        # flatten out image part (leaving patches & batch)
        patches = patches.flatten(2, 4).float()
        shape = patches.shape

        patches = (
            torch.cat([self._fc_list[i+1](patches[:, i]) for i in range(self.N)], dim=1)
            .reshape(shape[0], shape[1], self.hidden_size)
        ).to('cuda')

        # get embeddings (needs to match dimensionality of patch)
        embed = (torch.arange(1, shape[1]+1)
                    .repeat(shape[0], 1)
                    .unsqueeze(2)).to('cuda')
        # add embeddings to the patches
        patches_embed = torch.cat((embed, patches), 2)

        # add extra `class` - need one for each img in batch
        patches_embed = torch.cat(
            (self._class_embed
             .repeat(shape[0], 1)
             .unsqueeze(1), patches_embed), 1)

        return patches_embed.to('cuda')

class ImageTransformer(model.Model):
    def __init__(self, name, batch_size, lr, num_epochs, N_patches, hidden_size, n_transformer_layers, n_heads):
        super(ImageTransformer, self).__init__(name=name,
                                      batch_size=batch_size,
                                      lr=lr,
                                      num_epochs=num_epochs)
        self.N = N_patches
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_transformer_layers = n_transformer_layers

        # layers
        self._LP = LinearProjector(N_patches, hidden_size=hidden_size-1, batch_size=batch_size)
        self._LN = nn.LayerNorm(torch.Size((N_patches+1, hidden_size)))
        self._TEL = TransformerEncoderLayer(hidden_size, n_heads)
        self._TE = TransformerEncoder(self._TEL, n_transformer_layers, self._LN)
        self._fc = nn.Linear(hidden_size*(N_patches+1), 1)

    def forward(self, input):
        x = self._LP(input)
        x = self._TE(x)
        x = x.view(-1, self.hidden_size*(self.N+1))
        return self._fc(x)

class animeDataset(Dataset):
  def __init__(self, img_dir):
    self.img_dir = img_dir
    ids = glob.glob(f'{img_dir}/*')
    self.ids = [id.split('/')[-1] for id in ids]

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, id):
    img_path = f'{self.img_dir}/{self.ids[id]}'
    # this will return a tuple of (torch.tensor, array)
    img, label = pickle.load(open(img_path, 'rb'))
    label = torch.tensor(float(label))
    return img.to('cuda'), label.to('cuda')

def get_data_loaders(batch_size):
    ds = animeDataset('imgs')
    train_size = int(0.8*len(ds))
    val_size = len(ds) - train_size
    train, val = random_split(ds, [train_size, val_size])
    return DataLoader(train, batch_size=batch_size), DataLoader(val, batch_size=batch_size)

