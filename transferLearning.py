from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision
from torch.autograd import Variable

# NOTE: This is adapted from the torchvision datasets for
# CIFAR10 and CIFAR100, which can be found at
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
# This version allows a validation dataset to be created.
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    # validation examples will come from here
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, split='train',
                 transform=None, target_transform=None,
                 download=False, val_samples=1000):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split # train, val, or test

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.split in ['train', 'val']:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.val_data = self.train_data[-val_samples:]
            self.val_labels = self.train_labels[-val_samples:]
            self.train_data = self.train_data[:-val_samples]
            self.train_labels = self.train_labels[:-val_samples]
        elif self.split == 'test':
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            raise Exception('Unkown split {}'.format(self.split))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'train':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'val':
            img, target = self.val_data[index], self.val_labels[index]
        elif self.split == 'test':
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.split == 'train':
            return len(self.train_data)
        elif self.split == 'val':
            return len(self.val_data)
        elif self.split == 'test':
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)


import matplotlib.pyplot as plt

# You should implement these (softmax.py, twolayernn.py, convnet.py)
description='CIFAR-10 Example'
# Hyperparameters
lr = 0.0001
momentum = 0.9
weight_decay = 0.1
batch_size = 256
epochs = 20
model = 'mymodel'
no_cuda=True
seed=1
test_batch_size=1000
log_interval=10
cifar10_dir = 'data'
cuda = False
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# Load CIFAR10 using torch data paradigm
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# CIFAR10 meta data
n_classes = 10
im_size = (3, 32, 32)
# Subtract the mean color and divide by standard deviation. The mean image
# from part 1 of this homework was essentially a big gray blog, so
# subtracting the same color for all pixels doesn't make much difference.
# mean color of training images
cifar10_mean_color = [0.49131522, 0.48209435, 0.44646862]
# std dev of color across training images
cifar10_std_color = [0.01897398, 0.03039277, 0.03872553]
transform = transforms.Compose([
                 transforms.Resize(size=(224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize(cifar10_mean_color, cifar10_std_color),
            ])
# Datasets
train_dataset = CIFAR10(cifar10_dir, split='train', download=True,
                        transform=transform)
val_dataset = CIFAR10(cifar10_dir, split='val', download=True,
                        transform=transform)
test_dataset = CIFAR10(cifar10_dir, split='test', download=True,
                        transform=transform)
# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                 batch_size=batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset,
                 batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,
                 batch_size=batch_size, shuffle=True, **kwargs)

# def imshow(inp, title=None):
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array(cifar10_mean_color)
#     std = np.array(cifar10_std_color)
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # Pause a bit so that plots are updated

if model == 'mymodel':
    # model = models.mymodel.MyModel(im_size, args.hidden_dim, args.hidden_dim2,
    #                            args.kernel_size, args.kernel_size2, n_classes)
    model = torchvision.models.resnet152(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(2048, 10)
    print(model)
else:
    raise Exception('Unknown model {}'.format(model))
# cross-entropy loss function
criterion = F.cross_entropy
model.cpu()

#############################################################################
# TODO: Initialize an optimizer from the torch.optim package using the
# appropriate hyperparameters found in args. This only requires one line.
#############################################################################
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

def train(epoch):
    '''
    Train the model for one epoch.
    '''
    # Some models use slightly different forward passes and train and test
    # time (e.g., any model with Dropout). This puts the model in train mode
    # (as opposed to eval mode) so it knows which one to use.
    model.train()
    # train loop
    for batch_idx, batch in enumerate(train_loader):
        # prepare data
        images, targets = Variable(batch[0]), Variable(batch[1])
        print(images.shape)
        # imshow(images[0])
        if cuda:   
            images, targets = images.cuda(), targets.cuda()
        #############################################################################
        # TODO: Update the parameters in model using the optimizer from above.
        # This only requires a couple lines of code.
        #############################################################################
        loss = criterion(model(images), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        if batch_idx % log_interval == 0:
            val_loss, val_acc = evaluate('val', n_batches=4)
            train_loss = loss.data
            examples_this_epoch = batch_idx * len(images)
            epoch_progress = 100. * batch_idx / len(train_loader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(
                epoch, examples_this_epoch, len(train_loader.dataset),
                epoch_progress, train_loss, val_loss, val_acc))

def evaluate(split, verbose=False, n_batches=None):
    '''
    Compute loss on val or test data.
    '''
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    if split == 'val':
        loader = val_loader
    elif split == 'test':
        loader = test_loader
    for batch_i, batch in enumerate(loader):
        data, target = batch
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss += criterion(output, target, size_average=False).data
        # predict the argmax of the log-probabilities
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        n_examples += pred.size(0)
        if n_batches and (batch_i >= n_batches):
            break

    loss /= n_examples
    acc = 100. * correct / n_examples
    if verbose:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            split, loss, correct, n_examples, acc))
    return loss, acc


# train the model one epoch at a time


for epoch in range(1, epochs + 1):
    train(epoch)
evaluate('test', verbose=True)

# Save the model (architecture and weights)
model_save_name = 'transferLearning.pt'
path = F"/content/gdrive/My Drive/transferLearning.pt" 
torch.save(model, path)

# Later you can call torch.load(file) to re-load the trained model into python
# See http://pytorch.org/docs/master/notes/serialization.html for more details

