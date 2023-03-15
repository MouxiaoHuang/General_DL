export CUDA_VISIBLE_DEVICES=0,1

python3 train.py './exp_dir/resnet18_cifar10.py' --gpus 2
# python3 -m torch.distributed.launch --master_port 9999 --nproc_per_node=2 --use_env train.py ./exp_dir/resnet18_cifar10.py --distributed True