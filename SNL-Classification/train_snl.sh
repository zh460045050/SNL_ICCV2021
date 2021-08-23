$data_root=/mnt/nas/zhulei/datas/

CUDA_VISIBLE_DEVICES=0 python3 train_val.py --pretrained False \
                              --dataset imagenet \
                              --backbone resnet \
                              --arch 50 \
                              --data_dir $data_root \
                              --nl-type 'snl' \
                              --nl-nums 1 \
                              --checkpoints 'snl' \
                              --is_norm \
                              --is_sys


CUDA_VISIBLE_DEVICES=0 python3 train_val.py --pretrained False \
                              --dataset cifar100 \
                              --backbone preresnet \
                              --arch 56 \
                              --data_dir $data_root \
                              --nl-type 'snl' \
                              --nl-nums 1 \
                              --checkpoints 'snl' \
                              --is_norm \
                              --is_sys

CUDA_VISIBLE_DEVICES=0 python3 train_val.py --pretrained False \
                              --dataset cifar10 \
                              --backbone preresnet \
                              --arch 20 \
                              --data_dir $data_root \
                              --nl-type 'snl' \
                              --nl-nums 1 \
                              --checkpoints 'snl' \
                              --is_norm \
                              --is_sys