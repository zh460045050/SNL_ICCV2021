$data_root='path of the dataset'

CUDA_VISIBLE_DEVICES=0 python3 train_val.py --pretrained False \
                              --dataset cifar10 \
                              --backbone preresnet \
                              --arch 20 \
                              --data_dir $data_root \
                              --nl-type 'a2' \
                              --nl-nums 1 \
                              --checkpoints 'a2'

CUDA_VISIBLE_DEVICES=0 python3 train_val.py --pretrained False \
                              --dataset cifar10 \
                              --backbone preresnet \
                              --arch 20 \
                              --data_dir $data_root \
                              --nl-type 'nl' \
                              --nl-nums 1 \
                              --checkpoints 'nl'


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

CUDA_VISIBLE_DEVICES=0 python3 train_val.py --pretrained False \
                              --dataset cifar10 \
                              --backbone preresnet \
                              --arch 20 \
                              --data_dir $data_root \
                              --nl-type 'ns' \
                              --nl-nums 1 \
                              --checkpoints 'ns'

CUDA_VISIBLE_DEVICES=0 python3 train_val.py --pretrained False \
                              --dataset cifar10 \
                              --backbone preresnet \
                              --arch 20 \
                              --data_dir $data_root \
                              --nl-type 'cgnl' \
                              --nl-nums 1 \
                              --checkpoints 'cgnl'

CUDA_VISIBLE_DEVICES=0 python3 train_val.py --pretrained False \
                              --dataset cifar10 \
                              --backbone preresnet \
                              --arch 20 \
                              --data_dir $data_root \
                              --nl-type 'dnl' \
                              --nl-nums 1 \
                              --checkpoints 'dnl'

