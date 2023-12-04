import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from common.evaluate import *
import time
import os

def plot(title, train_loss, val_loss, epoch, folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    plt.figure()
    plt.title(f"Train vs Validation {title}")
    plt.plot(range(epoch), train_loss[:epoch], label="Train")
    plt.plot(range(epoch), val_loss[:epoch], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(f"{title}")
    plt.legend(loc='best')
    plt.savefig(f'{folder}/{epoch}')
    plt.close()

def train_model(model, train_loader, val_loader):
    torch.manual_seed(42)
    if not os.path.exists('training'):
        os.mkdir('training')
    num_epochs = model.num_epochs
    lr = model.lr
    logfile = open(model.str('logs'), 'a+')
    plot_folder = f'training/{model.name}_plots'
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = model.criterion

    # Set up some numpy arrays to store the training/test loss/erruracy
    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    best_loss = np.inf

    start = time.time()
    for epoch in range(num_epochs):
        total_loss = 0
        total_epoch = 0

        for batch in train_loader:
            outputs, loss, total_loss, total_epoch = calc_loss_per_batch(batch, model, criterion, total_loss, total_epoch)
            optimizer.zero_grad()
            loss = criterion(outputs, batch[1])
            loss.backward()
            optimizer.step()

        train_loss[epoch] = total_loss / len(train_loader)
        val_loss[epoch] = evaluate(model, val_loader)
        # scheduler.step(val_loss[epoch])
        print(f"Epoch {epoch}: Train loss {train_loss[epoch]} | Val loss {val_loss[epoch]}", file=logfile)

        if epoch%20 == 0 and epoch != 0:
            torch.save(model.state_dict(), model.str(epoch))
            plot('loss', train_loss, val_loss, epoch, plot_folder)
        if val_loss[epoch] < best_loss:
            best_loss = val_loss[epoch]
            torch.save(model.state_dict(), model.str(epoch))

    torch.save(model.state_dict(), model.str(epoch))
    print('Finished Training', file=logfile)
    end_time = time.time()
    elapsed_time = end_time - start
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time), file=logfile)
    # Write the train/test loss/err into CSV file for plotting later
    np.savetxt("{}_train_loss.csv".format(model.str(epoch)), train_loss)
    np.savetxt("{}_val_loss.csv".format(model.str(epoch)), val_loss)
    plot('loss', train_loss, val_loss, epoch, plot_folder)
    logfile.close()

class Model(nn.Module):
    def __init__(self, name, batch_size=64, num_epochs=200, lr=0.01):
        super(Model, self).__init__()
        self.criterion = nn.MSELoss()
        self.name = name
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

    def str(self, epoch):
        attrs_dict = self.__dict__
        attrs = [str(attrs_dict[key]) for key in attrs_dict.keys()
                 if ((key[0] != '_') and (type(attrs_dict[key]) in [str, int, float]))]
        return 'training/'+'_'.join([*attrs, str(epoch)])