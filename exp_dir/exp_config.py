model = dict(
    type='Classifier',
    encoder=dict(
        type='vit_base_patch16_224',
        pretrained=True,
        num_classes=10),
    test_cfg=dict(
        return_label=True,
        return_feature=False),
    train_cfg=dict(
        w_cls=1.0))
data_root = 'root_directory_of_data' # modify
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5],
    std=[255, 255, 255],
    to_rgb=False)
data = dict(
    train=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_files=['train_label.txt'],
        pipeline=dict(
            RandomFlip=dict(
                hflip_ratio=0.5,
                vflip_ratio=0),
            RandomRotate=dict(
                max_angle=18,
                rotate_ratio=0.5),
            RandomCrop=dict(
                crop_ratio=0.3,
                crop_range=(0.18, 0.18)),
            PhotoMetricDistortion=dict(
                hue=10,
                graying=0.1,
                contrast=(0.8, 1.2),
                brightness=(-16, 16),
                saturation=(0.9, 1.1),
                blur_sharpen=0.2,
                swap_channels=False),
            Resize=dict(scale=(224, 224)),
            Normalize=dict(**img_norm_cfg))),
    val=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_files=['val_label.txt'],
        pipeline=dict(
            Resize=dict(scale=(224, 224)),
            Normalize=dict(**img_norm_cfg))),
    test=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_files=['test_label.txt'],
        pipeline=dict(
            Resize=dict(scale=(224, 224)),
            Normalize=dict(**img_norm_cfg))),
    train_loader=dict(
        num_gpus=1,
        shuffle=True,
        samples_per_gpu=128,
        workers_per_gpu=8),
    test_loader=dict(
        num_gpus=1,
        shuffle=False,
        samples_per_gpu=256,
        workers_per_gpu=8))
log_cfg = dict(
    interval=20,
    filename=None,
    plog_cfg=dict(
        loss_types=['loss'],
        eval_types=['acc', 'auc', 'acer']))
eval_cfg = dict(
    interval=500,
    score_type='acc',
    tsne_cfg=dict(
        marks=None,
        filename='tsne.png'))
optim_cfg = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0)
sched_cfg = dict(
    type='CosineLR',
    gamma=0.01,
    warmup=2e3,
    total_epochs=120)
check_cfg = dict(
    interval=5e5,
    save_topk=3,
    load_from=None,
    resume_from=None)
total_epochs = 120
work_dir = 'exp_dir/exp_config_results'
gpu_ids = range(0, 1)
seed = 10
