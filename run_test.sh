export CUDA_VISIBLE_DEVICES=0

python3 test.py './exp_dir/resnet18_cifar10.py' --load_from './exp_dir/resnet18_cifar10/top1_model.pth'