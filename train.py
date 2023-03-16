import os
import time
import torch
import argparse
from torch.nn.parallel import DataParallel, DistributedDataParallel
from apis import Runner, build_models, build_datasets, build_dataloaders
from utils import Config, get_root_logger, seed_everywhere, init_distributed, get_rank, get_world_size


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config', help='path to train config file')
    parser.add_argument('--exp_dir', help='the dir to save logs and models')
    parser.add_argument('--load_from', help='the checkpoint file to load from')
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--distributed', type=bool, default=False, help='distributed')
    parser.add_argument('--gpus', type=int, default=1, help='the number of gpus to use')
    args = parser.parse_args()

    return args


def main():
    """main"""
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # set random seed
    if cfg.get('seed'):
        seed_everywhere(cfg.seed)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.exp_dir is not None:
        cfg.exp_dir = args.exp_dir

    if args.load_from is not None:
        cfg.check_cfg.load_from = args.load_from

    if args.resume_from is not None:
        cfg.check_cfg.resume_from = args.resume_from

    if args.distributed:
        init_distributed(args)

    cfg.gpu_ids = range(args.gpus)
    cfg.data.train_loader.num_gpus = args.gpus

    os.makedirs(os.path.expanduser(os.path.abspath(cfg.exp_dir)), exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.exp_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')
    cfg.log_cfg.filename = log_file

    model = build_models(cfg.model)
    dataset = build_datasets(cfg.data.train)

    if args.distributed:
        rank = get_rank()
        world_size = get_world_size()
        model = DistributedDataParallel(model.cuda(), device_ids=[args.gpus], find_unused_parameters=False)
        dataloader = build_dataloaders(cfg.data.train_loader, dataset, num_replicas=world_size, rank=rank)
    else:
        model = DataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        dataloader = build_dataloaders(cfg.data.train_loader, dataset)

    logger.info(f'Distributed training: {args.distributed}')
    logger.info(f'Train dataset class number: {len(dataset.groups)}')
    if len(dataset.groups) <= 10:
        logger.info(f'Train dataset: {dataset.groups}')

    runner = Runner(
        model,
        logger,
        exp_dir=cfg.exp_dir,
        log_cfg=cfg.log_cfg,
        eval_cfg=cfg.eval_cfg,
        optim_cfg=cfg.optim_cfg,
        sched_cfg=cfg.sched_cfg,
        check_cfg=cfg.check_cfg)

    if cfg.eval_cfg is not None:
        val_dataset = build_datasets(cfg.data.val)
        val_dataloader = build_dataloaders(cfg.data.test_loader, val_dataset)
        logger.info(f'Val dataset class number: {len(val_dataset.groups)}')
        if len(val_dataset.groups) <= 10:
            logger.info(f'Val dataset: {val_dataset.groups}')

        runner.val_dataloader = val_dataloader

    if cfg.get('step_cfg'):
        test_dataset = build_datasets(cfg.data.test)
        test_dataloader = build_dataloaders(cfg.data.test_loader, test_dataset)
        runner.test_dataloader = test_dataloader

        runner.train_step(dataloader, cfg)
    else:
        runner.train(dataloader, cfg)


if __name__ == '__main__':
    main()