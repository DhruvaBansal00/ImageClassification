import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim1, hidden_dim2, kernel_size1, kernel_size2, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.cnv1 = nn.Conv2d(im_size[0], hidden_dim1, kernel_size1, stride=1)
        self.cnv2 = nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size2, stride=1)
        output_size = hidden_dim2*int(((1 + 1 + im_size[-1] - kernel_size1 - kernel_size2)*(1 + 1 + im_size[-2] - kernel_size1 - kernel_size2))/9)
        # print("OUTPUT SIZE = ", output_size)
        self.fc1 = nn.Linear(output_size, n_classes)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
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
        # TODO: Implement the forward pass.
        #############################################################################
        cnv1_out = self.cnv1(images)
        relu1_out = F.relu(cnv1_out)
        # print(relu1_out.shape)
        cnv2_out = self.cnv2(relu1_out)
        relu2_out = F.relu(cnv2_out)
        # print(relu2_out.shape)
        pool = F.max_pool2d(relu2_out, 3, stride=3, padding=1)
        # print(pool.shape)
        image_size = pool.shape[-1]*pool.shape[-2]*pool.shape[-3]
        # print("IMAGE SIZE = ", image_size)
        fc_input = pool.view(-1, image_size)
        fc1_out = self.fc1(fc_input)
        scores = fc1_out
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

