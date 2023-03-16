exp_dir = 'exp_dir/resnet18_cifar10'
data_root = '/home/ssd5/huangmouxiao/database/CIFAR-10-images'
total_epochs = 200
seed = 42
model = dict(
    type='Classifier',
    encoder=dict(type='resnet18', pretrained=False, num_classes=10),
    test_cfg=dict(return_label=True, return_feature=False),
    train_cfg=dict(w_cls=1.0))
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[255, 255, 255], to_rgb=False)
data = dict(
    train=dict(
        type='CustomDataset',
        data_root=data_root,
        data_file='train_label.txt',
        pipeline=dict(
            RandomFlip=dict(hflip_ratio=0.5, vflip_ratio=0),
            Resize=dict(scale=(32, 32)),
            Normalize=dict(
                mean=[127.5, 127.5, 127.5], std=[255, 255, 255],
                to_rgb=False))),
    val=dict(
        type='CustomDataset',
        data_root=data_root,
        data_file='val_label.txt',
        pipeline=dict(
            Resize=dict(scale=(32, 32)),
            Normalize=dict(
                mean=[127.5, 127.5, 127.5], std=[255, 255, 255],
                to_rgb=False))),
    test=dict(
        type='CustomDataset',
        data_root=data_root,
        data_file='test_label.txt',
        pipeline=dict(
            Resize=dict(scale=(32, 32)),
            Normalize=dict(
                mean=[127.5, 127.5, 127.5], std=[255, 255, 255],
                to_rgb=False))),
    train_loader=dict(
        num_gpus=1, shuffle=True, samples_per_gpu=128, workers_per_gpu=8),
    test_loader=dict(
        num_gpus=1, shuffle=False, samples_per_gpu=256, workers_per_gpu=8))
log_cfg = dict(
    interval=20,
    filename=None,
    plog_cfg=dict(loss_types=['loss'], eval_types=['acc']))
eval_cfg = dict(
    interval=180,
    score_type='acc',
    tsne_cfg=dict(marks=None, filename='tsne.png'))
optim_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0)
sched_cfg = dict(type='CosineLR', gamma=0.01, warmup=0, total_epochs=total_epochs)
check_cfg = dict(
    interval=500000.0,
    save_topk=3,
    load_from=None,
    resume_from=None)
gpu_ids = range(0, 1)
