import torch
import torch.nn as nn

def evaluate(net, loader):
    """ Evaluate the network on the validation set.
    """
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        _, loss, total_loss, total_epoch = calc_loss_per_batch(data, net, criterion, total_loss, total_epoch)
    loss = total_loss / total_epoch
    return loss

def calc_loss_per_batch(data, net, criterion, total_loss, total_epoch):
    inputs, labels = data
    outputs = net(inputs)
    loss = torch.sqrt(criterion(outputs, labels.float()))
    total_loss += loss.item()
    total_epoch += len(labels)
    return outputs, loss, total_loss, total_epoch

def get_best_loss(model_name):
    import glob
    import numpy as np
    ls = glob.glob(f"training/{model_name}_val_loss.csv")
    # print(ls)
    minimums = [np.inf, 10, 100]

    for file in ls:
        with open(file, 'r') as f:
            nums = np.array(f.read().split()).astype(np.float_)
            if min(nums) < minimums[0]:
                minimums[0] = max(nums)
                minimums[1] = np.argmin(nums)
                minimums[2] = file

    print(minimums)
    return minimums
