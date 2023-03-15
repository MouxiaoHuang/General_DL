import os
import time
import torch
import argparse
from torch.nn.parallel import DataParallel
from utils import Config, get_root_logger, seed_everywhere
from apis import Runner, build_models, build_datasets, build_dataloaders


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config', help='path to train config file')
    parser.add_argument('--exp_dir', help='the dir to save logs and models')
    parser.add_argument('--load_from', help='the checkpoint file to load from')
    parser.add_argument('--result_flie', default='results.txt', help='the result file to save')
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

    cfg.gpu_ids = range(args.gpus)

    os.makedirs(os.path.expanduser(os.path.abspath(cfg.exp_dir)), exist_ok=True)
    cfg.dump(os.path.join(cfg.exp_dir, os.path.basename(args.config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.exp_dir, f'test_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    model = build_models(cfg.model)
    model = DataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    dataset = build_datasets(cfg.data.test)
    print(len(dataset))
    dataloader = build_dataloaders(cfg.data.test_loader, dataset)
    logger.info(f'Test dataset: {dataset.groups}')

    runner = Runner(
        model,
        logger,
        exp_dir=cfg.exp_dir,
        test_cfg=cfg.model.test_cfg,
        eval_cfg=cfg.eval_cfg,
        check_cfg=cfg.check_cfg)

    preds, labels = runner.test(dataloader, args.result_flie)
    print(preds.shape, labels.shape)


if __name__ == '__main__':
    main()
