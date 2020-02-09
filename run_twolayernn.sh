#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python3 -u train.py \
    --model twolayernn \
    --hidden-dim 40 \
    --epochs 20 \
    --weight-decay 0.01 \
    --momentum 0.9 \
    --batch-size 512 \
    --lr 0.00001 | tee twolayernn.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
