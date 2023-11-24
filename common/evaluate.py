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
    
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
        
    outputs = net(inputs)
    loss = torch.sqrt(criterion(outputs, labels.float()))
    total_loss += loss.item()
    total_epoch += len(labels)
    return outputs, loss, total_loss, total_epoch
