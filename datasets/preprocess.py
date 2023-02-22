import cv2
import numpy as np


class Preprocess(object):
    def __init__(self, pipeline):
        self.preprocess = []
        if pipeline is None:
            return
    
    def __call__(self, data):
        for process in self.preprocess:
            data = process(data)
        return data
    

class Normalize(object):
    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, data):
        img = data['img'].astype(np.float32)
        if self.to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        cv2.subtract(img, np.float64(self.mean.reshape(1, -1)), img)
        cv2.multiply(img, 1 / np.float64(self.std.reshape(1, -1)), img)
        data['img'] = img.transpose(2, 0, 1)
        return data