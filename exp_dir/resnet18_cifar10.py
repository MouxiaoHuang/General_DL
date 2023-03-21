exp_dir = 'exp_dir/resnet18_cifar10'
data_root = '/home/ssd5/huangmouxiao/database/cifar10'
total_epochs = 300
seed = 42
model = dict(
    type='Classifier',
    encoder=dict(type='resnet18', pretrained=False, num_classes=10, drop_rate=0.3),
    test_cfg=dict(return_label=True, return_feature=True),
    train_cfg=dict(w_cls=1.0, label_smoothing=0.1))
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    to_rgb=True)
data = dict(
    train=dict(
        type='CustomDataset',
        data_root=data_root,
        data_file='train_label.txt',
        pipeline=dict(
            RandomFlip=dict(hflip_ratio=0.5, vflip_ratio=0),
            RandomCrop=dict(crop_ratio=0.1, crop_range=(0.1, 0.1)),
            RandomRotate=dict(rotate_ratio=0.1, max_angle=8),
            PhotoMetricDistortion=dict(
                hue=10,
                graying=0.1,
                contrast=(0.8, 1.2),
                brightness=(-16, 16),
                saturation=(0.9, 1.1),
                blur_sharpen=0.2,
                swap_channels=False),
            Resize=dict(scale=(32, 32)),
            Normalize=dict(**img_norm_cfg))),
    val=dict(
        type='CustomDataset',
        data_root=data_root,
        data_file='val_label.txt',
        pipeline=dict(
            Resize=dict(scale=(32, 32)),
            Normalize=dict(**img_norm_cfg))),
    test=dict(
        type='CustomDataset',
        data_root=data_root,
        data_file='test_label.txt',
        pipeline=dict(
            Resize=dict(scale=(32, 32)),
            Normalize=dict(**img_norm_cfg))),
    train_loader=dict(
        num_gpus=1, shuffle=True, samples_per_gpu=128, workers_per_gpu=8),
    test_loader=dict(
        num_gpus=1, shuffle=False, samples_per_gpu=256, workers_per_gpu=8))
log_cfg = dict(
    interval=20,
    filename=None,
    plog_cfg=dict(loss_types=['loss', 'val_loss'], eval_types=['acc']))
eval_cfg = dict(
    interval=200,
    score_type='acc',
    tsne_cfg=dict(marks=None, filename='tsne.png'))
optim_cfg = dict(type='Adam', lr=0.01, weight_decay=0.001)
sched_cfg = dict(type='CosineLR', gamma=0.01, warmup=2000, total_epochs=total_epochs)
check_cfg = dict(
    interval=500000.0,
    save_topk=2,
    load_from=None,
    resume_from=None)
gpu_ids = range(0, 1)
