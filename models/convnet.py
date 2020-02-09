import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.hidden_size = hidden_dim
        self.cnv = nn.Conv2d(im_size[0], hidden_dim, kernel_size, stride=1)
        output_size = hidden_dim*(1 + im_size[-1] - kernel_size)*(1 + im_size[-2] - kernel_size)
        self.fc1 = nn.Linear(output_size, n_classes)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        cnv_out = self.cnv(images)
        image_size = self.hidden_size*cnv_out.shape[-1]*cnv_out.shape[-2]
        cnv_out = cnv_out.view(-1, image_size)
        fc1_out = self.fc1(F.relu(cnv_out))
        scores = fc1_out
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

