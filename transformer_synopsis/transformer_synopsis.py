import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
import pandas as pd
from common import model
from common import evaluate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformer_synopsis_model import *

batch_size=64
train, val, max_len = get_data_loaders('data_collection/data/balanced_animes_data_max_rank=5000.csv', batch_size)

# for samples, targets in train:
#     print(samples)
    
lr=0.01
num_epochs=100
input_size=max_len
hidden_size=128
num_class=1

ex = SynopsisTransformer('MySynTrans', batch_size, lr, num_epochs, input_size, hidden_size, num_class)