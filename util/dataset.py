import os

import tensorflow as tf
import numpy as np

from PIL import Image


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


class DataSet(object):
    def __init__(self, data_path, image_size=128):
        self.root_dir = data_path
        self.imgList = [f for f in os.listdir(data_path) if any(f.endswith(ext) for ext in IMG_EXTENSIONS)]
        self.imgList.sort()
        self.image_size = image_size
        self.images = np.zeros((len(self.imgList), image_size, image_size, 3)).astype(float)
        tf.logging.info('Loading dataset: {}'.format(data_path))
        for i in range(len(self.imgList)):
            image = Image.open(os.path.join(self.root_dir, self.imgList[i])).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image).astype(float)
            max_val = image.max()
            min_val = image.min()
            image = (image - min_val) / (max_val - min_val) * 2 - 1
            self.images[i] = image
        tf.logging.info('Data loaded, shape: {}'.format(self.images.shape))

    def data(self):
        return self.images

    def mean(self):
        return np.mean(self.images, axis=(0, 1, 2, 3))

    def to_range(self, low_bound, up_bound):
        min_val = self.images.min()
        max_val = self.images.max()
        return low_bound + (self.images - min_val) / (max_val - min_val) * (up_bound - low_bound)

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.imgList)