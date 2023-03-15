import cv2
import torch
import numpy as np


class Preprocess(object):
    def __init__(self, pipeline):
        self.preprocess = []
        if pipeline is None:
            return
        for k, args in pipeline.items():
            if k in ['RandomFlip', 'RandomRotate', 'RandomCrop',
                     'RandomErase', 'PhotoMetricDistortion', 'RandomGuassian',
                     'RandomMotionBlur', 'RandomZoomInOut', 'RandomBokehBlur']:
                self.preprocess.append(eval(f'{k}')(**args))
            elif k in ['Resize', 'Normalize']:
                continue
            else:
                raise NotImplementedError
        if 'Resize' in pipeline:
            self.preprocess.append(Resize(**pipeline['Resize']))
        if 'Normalize' in pipeline:
            self.preprocess.append(Normalize(**pipeline['Normalize']))
    
    def __call__(self, data):
        for process in self.preprocess:
            data = process(data)
        return data
    
class Normalize(object):
    """Normalize an image.

    Args:
        mean (tuple), std (tuple), to_rgb (bool)
    """
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
    
class Resize(object):
    """Resize an image.

    Args:
        scale (tuple): The target size of scale
    """

    def __init__(self,
                 scale):
        self.scale = scale

    def __call__(self, data):
        h, w = data['img'].shape[:2]
        w_scale = self.scale[0] / w
        h_scale = self.scale[1] / h
        data['img'] = cv2.resize(data['img'], self.scale, interpolation=cv2.INTER_LINEAR)
        return data
    
class RandomFlip(object):
    """Randomly flip an image.

    Args:
        hflip_ratio (float): The probability of flip an image horizontally.
        vflip_ratio (float): The probability of flip an image vertically.
    """

    def __init__(self,
                 hflip_ratio=0,
                 vflip_ratio=0):
        self.hflip_ratio = hflip_ratio
        self.vflip_ratio = vflip_ratio

    def horizontal_flip(self, data):
        h, w = data['img'].shape[:2]
        data['img'] = np.flip(data['img'], axis=1)
        return data

    def vertical_flip(self, data):
        h, w = data['img'].shape[:2]
        data['img'] = np.flip(data['img'], axis=0)
        return data

    def __call__(self, data):
        if np.random.uniform(0, 1) < self.hflip_ratio:
            data = self.horizontal_flip(data)
        if np.random.uniform(0, 1) < self.vflip_ratio:
            data = self.vertical_flip(data)
        return data

class RandomRotate(object):
    """Randomly roate an image.

    Args:
        rotate_ratio (float): The probobility to rotate an image.
        max_angle (int): Max angle of rotating an image.
    """

    def __init__(self,
                 max_angle=10,
                 rotate_ratio=0.0):
        self.max_angle = max_angle
        self.rotate_ratio = rotate_ratio
    
    def __call__(self, data):
        if np.random.uniform(0, 1) > self.rotate_ratio:
            return data
        h, w = data['img'].shape[:2]
        angle = np.random.randint(-self.max_angle, self.max_angle)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        data['img'] = cv2.warpAffine(data['img'], M, (w, h))
        return data
    
class RandomCrop(object):
    """Randomly applies crop to an image.

    Args:
        crop_ratio (float): The probability of applying crop blur.
        crop_range (tuple): The range of crop an image.
    """

    def __init__(self,
                 crop_ratio=0.0,
                 crop_range=(0.1, 0.1)):
        self.crop_ratio = crop_ratio
        self.crop_range = crop_range

    def __call__(self, data):
        if np.random.uniform(0, 1) > self.crop_ratio:
            return data
        h, w = data['img'].shape[:2]
        rx1 = np.random.randint(0, int(w * self.crop_range[0]))
        ry1 = np.random.randint(0, int(h * self.crop_range[1]))
        rx2 = np.random.randint(int((1-self.crop_range[0]) * w), w)
        ry2 = np.random.randint(int((1-self.crop_range[1]) * h), h)
        data['img'] = data['img'][ry1:ry2, rx1:rx2, :]
        return data

class RandomErase(object):
    """Random erase some region on image

    Args:
        erase_ratio (float): random erase probability.
        erase_shape (tuple): the shape of erase region, e.g. (0.05, 0.05) means
            shape of (0.05*w, 0.05*h).
        erase_fill (tuple): The value of pixel to fill in the erase regions.
            Default: (0, 0, 0)
    """

    def __init__(self,
                 erase_ratio=0,
                 erase_shape=(0.05, 0.05),
                 erase_fill=(0, 0, 0)):
        self.erase_ratio = erase_ratio
        self.erase_shape = erase_shape
        self.erase_fill = erase_fill

    def __call__(self, data):
        if np.random.uniform(0, 1) > self.erase_ratio:
            return data
        num_region = np.random.randint(1, 3)
        h, w = data['img'].shape[:2]
        for _ in range(num_region):
            ew = np.random.randint(0, int(self.erase_shape[0] * w))
            eh = np.random.randint(0, int(self.erase_shape[1] * h))

            x = np.random.randint(0, w - ew)
            y = np.random.randint(0, h - eh)

            data['img'][y:y+eh, x:x+ew, :] = self.erase_fill

        return data

class RandomGuassian(object):
    """Random add Guassian noise to an image

    Args:
        guassian_ratio (float): Random add probability.
        guassian_mean (float): Mean of gaussian.
        guassian_std (float): Standard deviation of gaussian.
    """

    def __init__(self,
                 guassian_ratio=0.0,
                 guassian_mean=0.0,
                 guassian_std=0.1):
        self.guassian_ratio = guassian_ratio
        self.guassian_mean = guassian_mean
        self.guassian_std = guassian_std
    
    def __call__(self, data):
        if np.random.uniform(0, 1) > self.guassian_ratio:
            return data
        epsilon = torch.randn(data['img'].shape).numpy()
        data['img'] = data['img'] + epsilon * self.guassian_std + self.guassian_mean
        return data

class RandomMotionBlur(object):
    """Randomly applies motion blur to an image.

    Args:
        motion_blur_ratio (float): The probability of applying motion blur.
        kernel_size (tuple): The size of the motion blur kernel.
        angle(int): The angle degree.
    """

    def __init__(self, motion_blur_ratio=0.0, kernel_size=(5,20), angle=90):
        self.motion_blur_ratio = motion_blur_ratio
        self.kernel_size = kernel_size
        self.angle = angle
    
    def __call__(self, data):
        if np.random.uniform(0, 1) > self.motion_blur_ratio:
            return data
        degree = np.random.randint(self.kernel_size[0], self.kernel_size[1])
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), self.angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        data['img'] = cv2.filter2D(data['img'], -1, motion_blur_kernel)
        return data
    
class RandomZoomInOut(object):
    """Randomly applies zoom-in and zoom-out to an image.

    Args:
        zoom_ratio (float): The probability of applying zoom-in and zoom-out.
        zoom_range (tuple): The range of zoom range.
    """

    def __init__(self, zoom_ratio=0.0, zoom_range=(112, 224)):
        self.zoom_ratio = zoom_ratio
        self.zoom_range = zoom_range
    
    def __call__(self, data):
        if np.random.uniform(0, 1) > self.zoom_ratio:
            return data
        zoom_target = int(np.random.uniform(self.zoom_range[0], self.zoom_range[1]))
        h, w = data['img'].shape[:2]
        data['img'] = cv2.resize(data['img'], (zoom_target, zoom_target), interpolation=cv2.INTER_LINEAR)
        data['img'] = cv2.resize(data['img'], (h, w), interpolation=cv2.INTER_LINEAR)
        return data
    
class RandomBokehBlur(object):
    """Randomly applies bokeh-like blur to an image.

    Args:
        bokeh_blur_ratio (float): The probability of applying bokeh-like blur.
        kernel_size (int): The size of the bokeh kernel.
        weights (float): The alpha / beta weights in cv2.addWeighted.
    """

    def __init__(self, bokeh_blur_ratio=0.0, kernel_size=(5, 20), weights=(1.5, -0.5)):
        self.bokeh_blur_ratio = bokeh_blur_ratio
        self.kernel_size = kernel_size
        self.weights = weights

    def __call__(self, data):
        if np.random.uniform(0, 1) > self.bokeh_blur_ratio:
            return data
        img_h, img_w, _ = data['img'].shape
        bokeh_kernel = np.zeros((img_h, img_w), dtype=np.float32)
        kernel_size_target = int(np.random.uniform(self.kernel_size[0], self.kernel_size[1]))
        cv2.circle(bokeh_kernel, (img_w//2, img_h//2), kernel_size_target, (1,1,1), -1, cv2.LINE_AA)
        bokeh_kernel /= bokeh_kernel.sum()
        bokeh_img = cv2.filter2D(data['img'], -1, bokeh_kernel)
        data['img'] = cv2.addWeighted(data['img'], self.weights[0], bokeh_img, self.weights[1], 0)
        return data
    
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    Args:
        hue (int): delta of hue.
        graying (float): the probability of graying.
        contrast (tuple): the range of contrast.
        brightness (tuple): the range of brightness.
        saturation (tuple): the range of saturation.
        blur_sharpen (float): the probability of blur or sharpen.
        swap_channels (bool): whether randomly swap channels
    """

    def __init__(self,
                 hue=18,
                 graying=0.0,
                 contrast=(0.5, 1.5),
                 brightness=(-32, 32),
                 saturation=(0.5, 1.5),
                 blur_sharpen=0.2,
                 swap_channels=False):
        self.hue = hue
        self.graying = graying
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.blur_sharpen = blur_sharpen
        self.swap_channels = swap_channels
    
    def __call__(self, data):
        img = data['img'].astype(np.float32)

        # randow graying
        if np.random.uniform(0, 1) < self.graying and (data['label'] != 0).any():  # only apply to attack
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis].repeat(3, 2)

        # random brightness
        if self.brightness is not None and np.random.randint(2):
            img += np.random.uniform(self.brightness[0], self.brightness[1])

        # random contrast
        if self.contrast is not None and np.random.randint(2):
            img *= np.random.uniform(self.contrast[0], self.contrast[0])

        # random blur or sharpen
        if np.random.uniform(0, 1) < self.blur_sharpen:
            if np.random.randint(2):
                img = cv2.GaussianBlur(img, (3, 3), 0)
            else:
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
                img = cv2.filter2D(img, -1, kernel=kernel)

        # convert color from BGR to HSV
        if self.saturation is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # random saturation
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(self.saturation[0], self.saturation[1])

            # random hue
            if np.random.randint(2):
                img[..., 0] += np.random.uniform(-self.hue, self.hue)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # randomly swap channels
        if self.swap_channels and np.random.randint(2):
            img = img[..., np.random.permutation(3)]

        data['img'] = np.clip(img, 0, 255)
        return data