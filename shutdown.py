import argparse
import os


if __name__ == '__main__':
    """Kill process using keyword of specific processes.
    Usage:
    1) ps -ef
        UID          PID    PPID  C STIME TTY          TIME CMD
        ...
        root        6666    6666  4 00:00 ?        00:00:00 python train.py ./exp_dir/resnet18_cifar10.py --gpus 2
        root        9999    9999  4 00:00 ?        00:00:00 python train.py ./exp_dir/resnet18_cifar10.py --gpus 2
        ...
    2) To kill the above two processes:
        python shutdown.py resnet18 (or resnet, resnet18_cifar10, cifar10, etc.)
    """
    parser = argparse.ArgumentParser(description='Shut down process with keyword')
    parser.add_argument('key', type=str, default='')
    args = parser.parse_args()
    os.system('ps -ef | grep ' + args.key + ' | grep -v grep | cut -c 9-16 | xargs kill -9')