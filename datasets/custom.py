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
        self.groups = self.set_group_flag()

    def _load_infos(self, data_file):
        data_infos = list()
        with open(os.path.join(self.data_root, self.data_file), 'r') as f:
            for line in f:
                line = line.strip('\n').rstrip().split()
                if len(line) == 2:
                    data_info = dict(
                        filename = line[0],
                        label = int(line[1])
                    )
                    data_infos.append(data_info)
                else:
                    raise NotImplementedError
        return data_infos
                    
    def set_group_flag(self):
        """Set flag according to label"""
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = self.data_infos[i]['label']
        return np.bincount(self.flag)
    
    def _rand_another(self, idx):
        """random select another index"""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    
    def _get_data(self, data_info):
        data = dict(
            img = cv2.imread(os.path.join(self.data_root, data_info['filename'])),
            label = np.ones((1,)).astype(np.int64) * data_info['label']
        )
        return data

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        while True:
            try:
                data = self._get_data(self.data_infos[idx])
            except:
                idx = self._rand_another(idx)
                continue
            break
        data = self.pipeline(data)
        return data