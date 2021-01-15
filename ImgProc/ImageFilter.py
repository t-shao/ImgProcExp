import numpy as np


class Filter(object):
    pass


class MedianFilter(Filter):
    def __init__(self, size, padding=False):
        self.size = size
        self.padding = padding

    def filter(self, image):
        if self.padding is True:
            raise NotImplementedError

        height = image.height
        width = image.width
        indexer = self.size // 2
        im = image.im
        new_im = np.zeros((height, width), dtype=np.uint8)

        if not self.padding:
            for i in range(height):
                for j in range(width):
                    if i <= indexer - 1 or i >= height - 1 - indexer or j <= indexer - 1 or j >= height - indexer - 1:
                        new_im[i, j] = im[i, j]
                    else:
                        new_im[i, j] = np.median(im[i - indexer:i + indexer + 1, j - indexer:j + indexer + 1])

        new = image.copy()
        new.im = new_im

        return new


# TODO: median_filter_fast


class MeanFilter(Filter):
    def __init__(self, size, padding=False):
        self.size = size
        self.padding = padding

    def filter(self, image):
        if self.padding is True:
            raise NotImplementedError

        height = image.height
        width = image.width
        indexer = self.size // 2
        im = image.im
        new_im = np.zeros((height, width), dtype=np.uint8)

        if not self.padding:
            for i in range(height):
                for j in range(width):
                    if i <= indexer - 1 or i >= height - 1 - indexer or j <= indexer - 1 or j >= height - indexer - 1:
                        new_im[i, j] = im[i, j]
                    else:
                        new_im[i, j] = np.mean(im[i - indexer:i + indexer + 1, j - indexer:j + indexer + 1])

        new = image.copy()
        new.im = new_im

        return new


# TODO: template operator
