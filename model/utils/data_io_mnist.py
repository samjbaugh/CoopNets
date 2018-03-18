from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

class DataSetMnist(object):
    def __init__(self, data_path, image_size=64, num_images=10000):
        data = input_data.read_data_sets(data_path, one_hot=True)
        self.image_size = image_size
        self.images = np.zeros((num_images, image_size, image_size, 3)).astype(float)
        print('Loading dataset: {}'.format(data_path))
        for i in range(len(self.imgList)):
            image = data.train.next_batch(1)[0]
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image).astype(float)
            max_val = image.max()
            min_val = image.min()
            image = (image - min_val) / (max_val - min_val) * 2 - 1
            self.images[i] = image
        print('Data loaded, shape: {}'.format(self.images.shape))

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