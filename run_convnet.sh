#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python3 -u train.py \
    --model convnet \
    --kernel-size 5  \
    --hidden-dim 64 \
    --epochs 20 \
    --weight-decay 0\
    --momentum 0 \
    --batch-size 1024 \
    --lr 0.0001 | tee convnet.log


# python3 -u train.py \
#     --model convnet \
#     --kernel-size 5  \
#     --hidden-dim 64 \
#     --epochs 20 \
#     --weight-decay 0.001\
#     --momentum 0.9 \
#     --batch-size 200 \
#     --lr 0.00002 >> convnet.log

# python3 -u train.py \
#     --model convnet \
#     --kernel-size 7  \
#     --hidden-dim 64 \
#     --epochs 20 \
#     --weight-decay 0.001\
#     --momentum 0.9 \
#     --batch-size 200 \
#     --lr 0.00002 >> convnet.log

# python3 -u train.py \
#     --model convnet \
#     --kernel-size 3  \
#     --hidden-dim 40 \
#     --epochs 20 \
#     --weight-decay 0.001\
#     --momentum 0.9 \
#     --batch-size 200 \
#     --lr 0.00002 >> convnet.log

# python3 -u train.py \
#     --model convnet \
#     --kernel-size 5  \
#     --hidden-dim 40 \
#     --epochs 20 \
#     --weight-decay 0.001\
#     --momentum 0.9 \
#     --batch-size 200 \
#     --lr 0.00002 >> convnet.log

# python3 -u train.py \
#     --model convnet \
#     --kernel-size 7  \
#     --hidden-dim 40 \
#     --epochs 20 \
#     --weight-decay 0.001\
#     --momentum 0.9 \
#     --batch-size 200 \
#     --lr 0.00002 >> convnet.log

# python3 -u train.py \
#     --model convnet \
#     --kernel-size 3  \
#     --hidden-dim 128 \
#     --epochs 20 \
#     --weight-decay 0.001\
#     --momentum 0.9 \
#     --batch-size 200 \
#     --lr 0.00002 >> convnet.log

# python3 -u train.py \
#     --model convnet \
#     --kernel-size 5  \
#     --hidden-dim 200 \
#     --epochs 20 \
#     --weight-decay 0.001\
#     --momentum 0.9 \
#     --batch-size 200 \
#     --lr 0.00002 >> convnet.log

# python3 -u train.py \
#     --model convnet \
#     --kernel-size 7  \
#     --hidden-dim 64 \
#     --epochs 20 \
#     --weight-decay 0.001\
#     --momentum 0.9 \
#     --batch-size 64 \
#     --lr 0.00001 >> convnet.log

# python3 -u train.py \
#     --model convnet \
#     --kernel-size 5  \
#     --hidden-dim 16 \
#     --epochs 20 \
#     --weight-decay 0.1\
#     --momentum 0.9 \
#     --batch-size 200 \
#     --lr 0.00001 >> convnet.log

# python3 -u train.py \
#     --model convnet \
#     --kernel-size 3  \
#     --hidden-dim 84 \
#     --epochs 20 \
#     --weight-decay 0.001\
#     --momentum 0.9 \
#     --batch-size 200 \
#     --lr 0.001 >> convnet.log

# python3 -u train.py \
#     --model convnet \
#     --kernel-size 5  \
#     --hidden-dim 64 \
#     --epochs 20 \
#     --weight-decay 0.9\
#     --momentum 0.9 \
#     --batch-size 200 \
#     --lr 0.00001 >> convnet.log

# python3 -u train.py \
#     --model convnet \
#     --kernel-size 5  \
#     --hidden-dim 128 \
#     --epochs 20 \
#     --weight-decay 0.1\
#     --momentum 0.9 \
#     --batch-size 200 \
#     --lr 0.00002 >> convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
