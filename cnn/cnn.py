from common import model
from common import evaluate
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import glob
import pickle
from torchvision import transforms

class CNN(model):
    def __init__(self, name, batch_size, lr, num_epochs):
        super(CNN, self).__init__(name=name,
                                  batch_size=batch_size,
                                  lr=lr,
                                  num_epochs=num_epochs)
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.fc1 = nn.Linear(10 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x

class ImageDataset(Dataset):
  def __init__(self, img_dir):
    self.img_dir = img_dir
    ids = glob.glob(f'{img_dir}/*')
    self.ids = [id.split('/')[-1] for id in ids]
    self.transform = transforms.RandomCrop(450, pad_if_needed=True)

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, id):
    img_path = f'{self.img_dir}/{self.ids[id]}'
    # this will return a tuple of (torch.tensor, array)
    img, label = pickle.loads(pickle.load(open(img_path, 'rb')))
    img = self.transform(img)
    label = torch.tensor(int(label))
    return img, label

def get_data_loaders(path_to_csv):
    df = pd.read_csv(path_to_csv)
    ds = animeDataset(df)
    train_size = int(0.8*len(ds))
    val_size = len(ds) - train_size
    train, val = random_split(ds, [train_size, val_size])
    return DataLoader(train, batch_size=32), DataLoader(val, batch_size=32)

from torch.utils.data import Subset, SubsetRandomSampler

def get_relevant_indices(dataset, classes, target_classes):
    """ Return the indices for datapoints in the dataset that belongs to the
    desired target classes, a subset of all possible classes.

    Args:
        dataset: Dataset object
        classes: A list of strings denoting the name of each class
        target_classes: A list of strings denoting the name of desired classes
                        Should be a subset of the 'classes'
    Returns:
        indices: list of indices that have labels corresponding to one of the
                 target classes
    """
    indices = []
    for i in range(len(dataset)):
        # Check if the label is in the target classes
        label_index = dataset[i][1] # ex: 3
        label_class = classes[label_index] # ex: 'cat'
        if label_class in target_classes:
            indices.append(i)
    return indices

def get_data_loader(batch_size, small=False):
    """ Loads images of cats and dogs, splits the data into training, validation
    and testing datasets. Returns data loaders for the three preprocessed datasets.

    Args:
        target_classes: A list of strings denoting the name of the desired
                        classes. Should be a subset of the argument 'classes'
        batch_size: A int representing the number of samples per batch

    Returns:
        train_loader: iterable training dataset organized according to batch size
        val_loader: iterable validation dataset organized according to batch size
        test_loader: iterable testing dataset organized according to batch size
        classes: A list of strings denoting the name of each class
    """

    ########################################################################
    # create dataset
    dataset = ImageDataset('/content/drive/MyDrive/APS360_Project/images/img')
    testset = ImageDataset('/content/drive/MyDrive/APS360_Project/images/test_set')

    # if small:
    #   # Define the desired subset size
    #   tsubset_size = 40
    #   vsubset_size = 10

    #   # Create a subset of the dataset
    #   train_subset_indices = torch.randperm(len(trainset))[:tsubset_size]
    #   valid_subset_indices = torch.randperm(len(validset))[:vsubset_size]
    #   trainset = Subset(trainset, train_subset_indices)
    #   validset = Subset(validset, valid_subset_indices)

    # # Get the list of indices to sample from
    indices = np.arange(len(dataset.ids))

    # # Split into train, test, and validation
    np.random.seed(1000) # Fixed numpy random seed for reproducible shuffling
    np.random.shuffle(indices)
    p80 = int(len(indices) * 0.8) #split at 80%

    # split into training and validation indices
    train_indices, val_indices = indices[:p80], indices[p80:]
    train_sampler = SubsetRandomSampler(train_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               num_workers=1, sampler=train_sampler)

    val_sampler = SubsetRandomSampler(val_indices)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              num_workers=1, sampler=val_sampler)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             num_workers=1)
    return train_loader, valid_loader, test_loader

###############################################################################
# Training
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return f"/content/drive/MyDrive/APS360_Project/images/model_training/{path}"

def normalize_label(labels):
    """
    Given a tensor containing 2 possible values, normalize this to 0/1

    Args:
        labels: a 1D tensor containing two possible scalar values
    Returns:
        A tensor normalize to 0/1 value
    """
    max_val = torch.max(labels)
    min_val = torch.min(labels)
    norm_labels = (labels - min_val)/(max_val - min_val)
    return norm_labels

if __name__ == '__main__':
    # Set the random seed for reproducible experiments
    torch.manual_seed(1000)

    # Get the dataset
    train_loader, valid_loader, test_loader = get_data_loader(batch_size=32, small=False)

    # Create the model
    model = CNN(name='cnn', batch_size=32, lr=0.001, num_epochs=100)

    # Train the model
    train_model(model, train_loader, valid_loader)

    # Evaluate the model on test set
    evaluate(model, test_loader)