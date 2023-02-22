import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from datasets.preprocess import Preprocess


class CustomDataset(Dataset):
    def __init__(self,
                 data_root,
                 data_file,
                 pipeline=None) -> None:
        super(CustomDataset, self).__init__()
        self.data_root = data_root
        self.data_file = data_file
        self.pipeline = Preprocess(pipeline)
        self.data_infos = self._load_infos(self.data_file)

    def _load_infos(self, data_file):
        data_infos = list()
        with open(os.path.join(self.data_root, self.data_file), 'r') as f:
            for line in f:
                line = line.strip('\n').rstrip().split()
                # set the label as one if the dataset has no groundtruth
                if len(line) == 1:
                    data_infos = dict(
                        filename = line[0],
                        label = -1
                    )
                elif len(line) == 2:
                    data_infos = dict(
                        filename = line[0],
                        label = int(line[1])
                    )
                else:
                    assert NotImplementedError
    
    def _get_data(self, data_info):
        data = dict(
            img = cv2.imread(data_info['imgname']),
            label = np.ones((1,)).astype(np.int64) * data_info['label']
        )

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        data = self._get_data(self.data_infos[idx])
        data = self.pipeline(data)
        return data