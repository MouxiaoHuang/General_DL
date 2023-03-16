## General Deep Learning

***A general deep learning project that can be easily transferred to other specific tasks.***

#### Basic environment:
---
`torch` and `timm`

#### Structure of the Repository
---
- `apis/`
  - `builder.py`: Builds datasets, dataloaders, models, optimizers, schedulers, and more.
  - `evaluator.py`: Evaluates metrics.
  - `runner.py`: Handles training, validation, and inference.
  - `sampler.py`: Provides samplers for balanced, distributed, and other purposes.
  - `visualizer.py`: Offers visualization tools such as TSNE, metrics, and more.
- `datasets/`
  - `custom.py`: Defines custom datasets for images.
  - `preprocess.py`: Preprocesses input data.
- `models/`
  - `backbones/`: Defines networks of backbones (encoders / feature extractors, etc.).
  - `losses/`: Defines loss functions.
  - `model/`: Defines complete models (e.g., classifiers), including backbones, heads, and losses.
- `utils/`
  - `config.py`: Interprets configuration files.
  - `dist.py`: Implements distributed training.
  - `fileio.py`:  Loads and dumps files (e.g., json, pickle, txt, csv).
  - `logger.py`: Initializes logger.
  - `seed.py`: Sets random seed.
  - `gen_imglist.py`: Generates imagelists for datasets.
- `shutdown.py`: Kills processes with keywords.
- `train.py` and `test.py`: Main files for training (validation) and inference.
- `run.sh` and `run_test.sh`: Scripts for experiments.
- `exp_dir/`: Experimental directory including configuration files, logs, checkpoints, and more.

#### Instruction for usage
---
- Prepare dataset
  ```python
  data_root/
    - train/
    - val/
    - test/
    - {train, val, test}_label.txt (format: relative_path label)
  ```
- Training
  ```python
  sh run.sh
  # or
  nohup sh run.sh>train.out 2>&1 &
  ```
  Training logs and checkpoints will be saved in `./exp_dir/resnet18_cifar10`
- Inference
  ```python
  sh run_test.sh
  # or
  nohup sh run_test.sh>test.out 2>&1 &
  ```
  Inference logs and results will be saved in `./exp_dir/resnet18_cifar10`
- Extensions can be added to the existing structure as needed.