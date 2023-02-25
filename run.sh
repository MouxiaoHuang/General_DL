export CUDA_VISIBLE_DEVICES=1,2,3,4

python train.py './exp_dir/exp_config.py' --gpus 4
# python -m torch.distributed.launch --master_port 9999 --nproc_per_node=4 --use_env train.py ./exp_dir/exp_config.py --distributed True
