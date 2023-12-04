import torch
import torch.nn as nn

import glob
import numpy as np

import os

import math

def evaluate(net, loader):
    """ Evaluate the network on the validation set.
    """
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        _, loss, total_loss, total_epoch = calc_loss_per_batch(data, net, criterion, total_loss, total_epoch)
    loss = total_loss / len(loader)
    return loss

def calc_loss_per_batch(data, net, criterion, total_loss, total_epoch):
    inputs, labels = data
    
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
        
    outputs = net(inputs)
    loss = torch.sqrt(criterion(outputs, labels.float()))
    # print(loss)
    total_loss += loss.item()
    total_epoch += len(labels)
    return outputs, loss, total_loss, total_epoch

def get_best_loss(path):
    os.chdir('C:\\Users\\Luke Yang\\Documents\\100_Luke\\100_School\\130_University_of_Toronto\\2023_2024\\APS360\\Anime-popularity-predictor')
    ls = glob.glob(f"{path}/*_val_loss.csv")
    # print(ls)
    minimums = [np.inf, 10, 100]

    for file in ls:
        with open(file, 'r') as f:
            nums = np.array(f.read().split()).astype(np.float_)
            if min(nums) < minimums[0]:
                minimums[0] = min(nums)
                minimums[1] = np.argmin(nums)
                minimums[2] = file

    print(minimums)
    return minimums

# def correct_loss():
#     os.chdir('C:\\Users\\Luke Yang\\Documents\\100_Luke\\100_School\\130_University_of_Toronto\\2023_2024\\APS360\\Anime-popularity-predictor')
#     ls = glob.glob(f"training_corrected/*_val_loss.csv")
#     for file in ls:
#         # 848 should be the total number of validation data points
#         # 4th index should be the batch size, but double check
#         num_of_batch = math.ceil(848 / int(file.split("_")[4]))
#         with open(file, 'r') as f:
#             nums = np.array(f.read().split()).astype(np.float_)
#             for i in range(len(nums)):
#                 nums[i] = (nums[i] * 848) / (num_of_batch)
#         # name where you want to write the saved files
#         # make sure you don't overwrite the original files because you
#         with open(<DESTINATION_FILE>, 'w') as f:
#             for num in nums:
#                 f.write(str(num) + '\n')
#     return

def create_params():
  # n_hid, activation; opt, lr, bs, max_ep
  rnd = np.random.RandomState()

  batch_size = rnd.randint(1, 512)
  lr = rnd.uniform(low=0.0001, high=0.10)
  hidden_size = rnd.randint(128, 512)


  return (batch_size, lr, hidden_size)