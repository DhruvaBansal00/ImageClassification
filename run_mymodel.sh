#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
# python3 -u train.py \
#     --model mymodel \
#     --kernel-size 5 \
#     --kernel-size2 3 \
#     --hidden-dim 64 \
#     --hidden-dim2 128 \
#     --epochs 10 \
#     --weight-decay 0.1 \
#     --momentum 0.9 \
#     --batch-size 512 \
#     --lr 0.0001 | tee mymodel.log
# RUNNING RN


python3 -u train.py \
    --model mymodel \
    --kernel-size 5 \
    --kernel-size2 5 \
    --hidden-dim 64 \
    --hidden-dim2 64 \
    --epochs 10 \
    --weight-decay 0.1 \
    --momentum 0.9 \
    --batch-size 50 \
    --lr 0.0001 | tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
