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
from torchvision.models import mnasnet0_75, MNASNet0_75_Weights, shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights, efficientnet_b0, EfficientNet_B0_Weights

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from common import model
from common import evaluate

class ImageDataset(Dataset):
  def __init__(self, img_dir, preprocess=None):
    if img_dir.endswith('/'):
        img_dir = img_dir[:-1]
    self.img_dir = img_dir
    ids = glob.glob(img_dir+"/*")
    ids = [id.replace("\\", "/") for id in ids]
    self.ids = [id.split('/')[-1] for id in ids]
    if preprocess is None:
        elf.transform = transforms.RandomCrop(450, pad_if_needed=True)
    else:
        self.transform = preprocess

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

def normalize_label(label):
    max_val = 4992 # That is the max rank
    min_val = 1 # min rank
    norm_label = (label - min_val)/(max_val - min_val)
    return norm_label

def get_data_loader(batch_size, preprocess=None):
    # Get the datasets
    dataset = ImageDataset('./data_collection/img_data/images', preprocess=preprocess) 
    testset = ImageDataset('./data_collection/img_data/test_images', preprocess=preprocess)
    
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

def get_model_name(name, batch_size, learning_rate, num_epochs, epoch):
    path = f"./training/{name}_{batch_size}_{learning_rate}_{num_epochs}_{epoch}"
    return path

if __name__ == '__main__':
    # Get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    
    # Setup models
    # Pretrained models to try
    # 'shufflenet_v2_x1_5', 'mnasnet0_75', 'efficientnet_b0'
    # shufflenet_v2_x1_5, mnasnet0_75, efficientnet_b0
    # ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1, MNASNet0_75_Weights.IMAGENET1K_V1, EfficientNet_B0_Weights.IMAGENET1K_V1
    pretrained_models_names = ['shufflenet_v2_x1_5', 'mnasnet0_75']
    pretrained_models = [shufflenet_v2_x1_5, mnasnet0_75]
    pretrained_weights = [ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1, MNASNet0_75_Weights.IMAGENET1K_V1]
    already_trained = [shufflenet_v2_x1_5, mnasnet0_75]
    # Model: Top-1 Accuracy, Top-5 Accuracy, # Parameters
    # EfficientNet_B0_Weights.IMAGENET1K_V1: 77.692, 93.532, 5.3M
    # ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1: 72.996, 91.086, 3.5M
    # MNASNet0_75_Weights.IMAGENET1K_V1: 71.18, 90.496, 3.2M
    
    # Hyperparameters to tune
    fine_tuning_options = [True, False]
    
    # Training parameters
    nums_epochs = 100
    batch_size = 128
    lr = 0.001
    
    for fine_tuning in fine_tuning_options:
        for i in range(len(pretrained_models)):
            if pretrained_models[i] in already_trained and fine_tuning == True:
                continue
            print(f"Training model: {pretrained_models_names[i]} with fine tuning: {fine_tuning}")
            # Import pretrained model and weights
            weights = pretrained_weights[i]
            cur_model = pretrained_models[i](weights=weights)
            model_name = pretrained_models_names[i]
            # If not fine tuning, freeze all layers except the last one
            if fine_tuning == False:
                for param in cur_model.parameters():
                    param.requires_grad = False
            # Modify classifier for regression
            if model_name == 'shufflenet_v2_x1_5':
                num_features = cur_model.fc.in_features
                cur_model.fc = nn.Linear(num_features, 1)
            else:
                num_features = cur_model.classifier[1].in_features
                cur_model.classifier[1] = nn.Linear(num_features, 1)
            cur_model = cur_model.to(device) # Move model to the GPU
            #summary(cur_model, (3, 224, 224))
            # Prepare data
            preprocess = weights.transforms()
            train_loader, valid_loader, test_loader = get_data_loader(batch_size=batch_size, preprocess=preprocess)
            name = model_name if fine_tuning else f"{model_name}_frozen"
            pretrained_params = {'lr': lr, 'num_epochs': nums_epochs, 'batch_size': batch_size, 'name': name}
            # Train the model
            torch.cuda.empty_cache()
            model.train_model(cur_model, train_loader, valid_loader,
                              pretrained=True, pretrained_params=pretrained_params)
    # Trained efficient net with: exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # Changed to step_size=40, gamma=0.5 for other models
    
    
    
    
    # Apply it to the input image
    # batch = preprocess(img).unsqueeze(0)
    # # Use the model and print the predicted category
    # prediction = model(batch).squeeze(0).softmax(0)
    # class_id = prediction.argmax().item()
    # score = prediction[class_id].item()
    # category_name = weights.meta["categories"][class_id]
    # print(f"{category_name}: {100 * score:.1f}%")
    
    # visualize_model(model)

    # img, label = pickle.loads(pickle.load(open('./data_collection/img_data/images/0', 'rb')))
    # transform = transforms.RandomCrop(450, pad_if_needed=True)
    # img = transform(img)
    # print(img.shape)
    # outputs = cnn(img)
    # print(outputs)
    
    # Load best model
    # name = "shufflenet_v2_x1_5"
    # bs = 128
    # lr = 0.001
    # num_epochs = 100
    # best_epoch = 37
    # cur_model = shufflenet_v2_x1_5(weights=None)
    # if name == 'shufflenet_v2_x1_5':
    #     num_features = cur_model.fc.in_features
    #     cur_model.fc = nn.Linear(num_features, 1)
    # else:
    #     num_features = cur_model.classifier[1].in_features
    #     cur_model.classifier[1] = nn.Linear(num_features, 1)
    # model_path = get_model_name(name=name, batch_size=bs,
    #                             learning_rate=lr, num_epochs=num_epochs, epoch=best_epoch)
    # state = torch.load(model_path)
    # cur_model.load_state_dict(state)
    # cur_model = cur_model.to(device) # Move model to the GPU
    # # Evaluate the model on test set
    # test_loss = evaluate.evaluate(cur_model, test_loader, pretrained=True)
    # print(f"Test loss: {test_loss}")
