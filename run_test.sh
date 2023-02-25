export CUDA_VISIBLE_DEVICES=0

python test.py './exp_dir/exp_config.py' --load_from './exp_dir/exp_config_results/top1_model.pth'