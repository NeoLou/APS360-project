import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchsummary import summary
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from common import model
from common import evaluate
class CNN(model.Model):
    def __init__(self, num_groups, num_neurons_fc, name, batch_size, num_epochs, lr):
        super(CNN, self).__init__(name=name,
                                      batch_size=batch_size,
                                      lr=lr,
                                      num_epochs=num_epochs)
        assert num_groups >= 1, f"You must have at least 2 convolutional groups"

        # create conv layers
        # this generates list of (2 * num_groups) evenly spaced numbers from 8 to 64
        num_filter_list = np.ceil(np.linspace(8, 64, 2*num_groups)).astype(int)
        self.conv_layers, self.map_size = self.create_conv_layers(num_filter_list)
        # create fully connected layers
        self.fc = nn.Linear(self.map_size**2*64, num_neurons_fc)
        # create regression layer
        self.regression = nn.Linear(num_neurons_fc, 1)

    def create_conv_layers(self, num_filter_list):
        #generate a list of that has the number of filters that each layer will have
        groups = nn.ModuleList()

        j = 0
        for i in range(len(num_filter_list)//2):
            if i == 0:
                # Conv2d(in_ch, out_ch, k_size)
                # Output dims after conv1 stays the same
                conv1 = nn.Conv2d(3, num_filter_list[j], kernel_size=3, padding=1)
                # Output dims after conv2 is: floor((input - 1)/2) + 1
                conv2 = nn.Conv2d(num_filter_list[j], num_filter_list[j+1], 
                                  kernel_size=3, padding=1, stride=2)
                groups.extend(deepcopy([conv1, conv2]))
                j += 2
            else:
                # Output dims after conv1 stays the same
                conv1 = nn.Conv2d(num_filter_list[j-1], num_filter_list[j], 
                                  kernel_size=3, padding=1)
                # Output dims after conv2 is: floor((input - 1)/2) + 1
                conv2 = nn.Conv2d(num_filter_list[j], num_filter_list[j+1],
                                  kernel_size=3, padding=1, stride=2)
                # Default stride is kernel_size, so ouput dims: floor((input - 3)/2) + 1
                maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
                if num_groups < 5:
                    groups.extend(deepcopy([conv1, conv2, maxpool]))
                else:
                    groups.extend(deepcopy([conv1, conv2]))
                j += 2
            if j >= len(num_filter_list):
                # figure out how to squash everything down to 1x64
                # loop through group list to figure out how big we have to make the next layer
                map_size = 450 # start from input image size
                for iter in range(0, num_groups):
                    if num_groups < 5:
                        map_size = ((map_size - 1) // 2) + 1
                        if iter != 0:
                            map_size = ((map_size - 3) // 2) + 1
                    else:
                        map_size = ((map_size - 1) // 2) + 1
                break
        return groups, map_size

    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            skip = 0
            if (i+1) % 3 == 0:
                # add a skip connection
                curr_forward = layer(x)
                x = F.leaky_relu(curr_forward) + skip
                skip = F.leaky_relu(curr_forward)
            else:
                x = F.leaky_relu(layer(x))
        x = x.view(-1, self.map_size**2*64)
        x = F.relu(self.fc(x))
        x = self.regression(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x
class ImageDataset(Dataset):
  def __init__(self, img_dir):
    if img_dir.endswith('/'):
        img_dir = img_dir[:-1]
    self.img_dir = img_dir
    ids = glob.glob(img_dir+"/*")
    ids = [id.replace("\\", "/") for id in ids]
    self.ids = [id.split('/')[-1] for id in ids]
    self.transform = transforms.RandomCrop(450, pad_if_needed=True)

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, id):
    img_path = f'{self.img_dir}/{self.ids[id]}'
    # this will return a tuple of (torch.tensor, array)
    img, label = pickle.loads(pickle.load(open(img_path, 'rb')))
    img = self.transform(img)
    label = torch.tensor(float(label))
    label = normalize_label(label)
    return img, label

def get_data_loader(batch_size):
    dataset = ImageDataset('./data_collection/img_data/images') 
    testset = ImageDataset('./data_collection/img_data/test_images')
    
    # Get the list of indices to sample from
    indices = np.arange(len(dataset))
    
    # # Split into train, test, and validation
    np.random.seed(1000) # Fixed numpy random seed for reproducible shuffling
    np.random.shuffle(indices)
    p80 = int(len(indices) * 0.8) #split at 60%
    
    # split into training and validation indices
    train_indices, val_indices = indices[:p80], indices[p80:]
    train_sampler = SubsetRandomSampler(train_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              num_workers=2, sampler=train_sampler)

    val_sampler = SubsetRandomSampler(val_indices)
    valid_loader = DataLoader(dataset, batch_size=batch_size,
                              num_workers=2, sampler=val_sampler)

    test_loader = DataLoader(testset, batch_size=batch_size,
                             num_workers=2)

    return train_loader, valid_loader, test_loader

def get_model_name(models_dir, name, batch_size, learning_rate, epoch):
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return f"{models_dir}{path}"

def normalize_label(label):
    max_val = 4992 # That is the max rank
    min_val = 1 # min rank
    norm_label = (label - min_val)/(max_val - min_val)
    return norm_label

if __name__ == '__main__':
    # Hyperparameters to tune
    choice_num_groups = [6]
    choice_num_epochs = [120]
    choice_neurons_fc = [20]
    choice_lr = [0.0001]
    
    # Set training parameters
    batch_size = 32
    #lr = 0.001
    #num_epochs = 40
    #num_groups = 4
    #num_neurons_fc = 20
    
    # Get data loaders
    train_loader, valid_loader, test_loader = get_data_loader(batch_size=batch_size)

    # Create the model
    for num_groups in choice_num_groups:
        for num_epochs in choice_num_epochs:
            for num_neurons_fc in choice_neurons_fc:
                for lr in choice_lr:
                    name = f"CNN_{num_groups}_{num_neurons_fc}"
                    cnn = CNN(num_groups=num_groups, num_neurons_fc=num_neurons_fc, name=name,
                            batch_size=batch_size, num_epochs=num_epochs, lr=lr)
                    # Train the model
                    #model.train_model(cnn, train_loader, valid_loader)
                    #summary(cnn, (3, 450, 450))
    # img, label = pickle.loads(pickle.load(open('./data_collection/img_data/images/0', 'rb')))
    # transform = transforms.RandomCrop(450, pad_if_needed=True)
    # img = transform(img)
    # print(img.shape)
    # outputs = cnn(img)
    # print(outputs)

    # Load best model
    # CNN_4_20_64_0.001_200
    # Epoch 171: Train loss 0.10238730809036291 | Val loss 0.18527020194700786
    num_groups = 4
    num_neurons_fc = 20
    batch_size = 64
    lr = 0.001
    num_epochs = 200
    best_epoch = 171
    name = f"CNN_{num_groups}_{num_neurons_fc}"
    
    # Create empty model
    cnn = CNN(num_groups=num_groups, num_neurons_fc=num_neurons_fc, name=name,
                            batch_size=batch_size, num_epochs=num_epochs, lr=lr)
    # Load best model
    print(f"Loading model {name} at epoch {best_epoch}")
    state = torch.load(cnn.str(best_epoch))
    cnn.load_state_dict(state)
    # Evaluate the model on test set
    print(f"Evaluating model {name} on test set")
    test_loss = evaluate.evaluate(cnn, test_loader)
    print(f"Test loss: {test_loss}") # Test loss: 0.27954493356602533