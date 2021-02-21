
CUDA_VISIBLE_DEVICES=$1 python train_one.py \
    --model $2 \
    --gpu_id $1 \
    --num_epochs 200 \
    --schedule 60 120 160 \
    --seed 0 \
    --dataset CIFAR100
