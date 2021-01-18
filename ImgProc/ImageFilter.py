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


class Kernel(object):
    """
        Create a convolution kernel.

        :param size: Kernel size.
        :param kernel: A numpy array containing kernel weights.
        :param scale: Scale factor. If given, the result for each pixel is
                        divided by this value.  The default is the sum of the
                        kernel weights.
        :param offset: Offset. If given, this value is added to the result,
                        after it has been divided by the scale factor.
    """
    def __init__(self, size, kernel, scale=None, offset=0):
        if scale is None:
            # default scale is sum of kernel
            scale = np.sum(kernel) if np.sum(kernel) != 0 else 1
        if kernel.shape != size:
            raise ValueError
        self.filterargs = size[0], scale, offset, kernel

    @property
    def size(self):
        return self.filterargs[0]

    @property
    def scale(self):
        return self.filterargs[1]

    @property
    def offset(self):
        return self.filterargs[2]

    @property
    def kernel(self):
        return self.filterargs[3]

    def _convolve(self, im):
        kernel = self.kernel
        kernel_size = self.size

        h = int(im.shape[0])
        w = int(im.shape[1])
        # padding
        padding = int((kernel_size - 1) / 2)
        padded_h = int(h + 2 * padding)
        padded_w = int(w + 2 * padding)

        im_padded = np.zeros((padded_h, padded_w))
        im_padded[padding:padding + h, padding:padding + w] = im[:, :]

        im_res = np.zeros(im.shape)
        for i in range(padding, padding + h):
            for j in range(padding, padding + w):
                im_res[i - padding][j - padding] = np.sum(
                    im_padded[i - padding:i + padding + 1, j - padding:j + padding + 1] * kernel, dtype=np.uint8)

        return im_res

    def filter(self, image):
        im = image.im
        new = image.copy()
        new.im = np.uint8(self._convolve(im) / self.scale + self.offset)

        return new


class Sobel(Kernel):
    def __init__(self, mode):
        if mode not in ['v', 'h']:
            raise ValueError

        size = (3, 3)
        if mode == 'v':
            kernel = np.array([[-1,  0,  1],
                               [-2,  0,  2],
                               [-1,  0,  1]])
        else:
            kernel = np.array([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]])

        super().__init__(size, kernel)


class Prewitt(Kernel):
    def __init__(self, mode):
        if mode not in ['v', 'h']:
            raise ValueError

        size = (3, 3)
        if mode == 'v':
            kernel = np.array([[-1,  0,  1],
                               [-1,  0,  1],
                               [-1,  0,  1]])
        else:
            kernel = np.array([[-1, -1, -1],
                               [ 0,  0,  0],
                               [ 1,  1,  1]])

        super().__init__(size, kernel)


# TODO: Robert operator is not properly functional yet
class Robert(Kernel):
    def __init__(self, mode):
        if mode not in [45, 135]:
            raise ValueError

        size = (2, 2)
        if mode == 45:
            kernel = np.array([[ 1,  0],
                               [ 0, -1]])
        else:
            kernel = np.array([[ 0,  1],
                               [-1,  0]])

        super().__init__(size, kernel)


class Laplacian(Kernel):
    def __init__(self):
        size = (3, 3)
        kernel = np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]])

        super().__init__(size, kernel)
