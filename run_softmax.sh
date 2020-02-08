#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python3 -u train.py \
    --model softmax \
    --epochs 10 \
    --weight-decay 0.99 \
    --momentum 0.9 \
    --batch-size 512 \
    --lr 0.001 | tee softmax.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
