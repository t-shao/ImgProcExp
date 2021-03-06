import numpy as np
from PIL import Image as PILImage
from matplotlib import pyplot as plt

from .Bmp import Bmp

# color modes
COLOR_UNDEF = None
COLOR_1 = '1'
COLOR_L = 'L'
COLOR_RGB = 'RGB'
COLOR_YCbCr = 'YCbCr'
COLOR_CMYK = 'CMYK'
COLOR_YIQ = 'YIQ'
COLOR_HSI = 'HSI'
COLOR_XYZ = 'XYZ'

# color conversion matrix
CLR_CVT_RGB2L = np.array([299, 587, 114])
CLR_CVT_FROM_RGB_W = {
    COLOR_YCbCr: np.array([
        [.299, .587, .114],
        [-.168736, -.331264, .5],
        [.5, -.418688, -.081312]
    ]),
    COLOR_YIQ: np.array([
        [.299, .587, .114],
        [.596, -.274, -.322],
        [.211, -.523, .312]
    ]),
    COLOR_XYZ: np.array([
        [.490, .310, .2],
        [.177, .813, .011],
        [0, .01, .99]
    ])
}
CLR_CVT_FROM_RGB_B = {
    COLOR_YCbCr: np.array([
        0, 128, 128
    ]),
    COLOR_YIQ: np.array([
        0, 0, 0
    ]),
    COLOR_XYZ: np.array([
        0, 0, 0
    ])
}
CLR_CVT_TO_RGB_W = {
    COLOR_YCbCr: np.array([
        [1, 0, 1.402],
        [1, -.344136, -.714136],
        [1, 1.772, 0]
    ]),
    COLOR_YIQ: np.linalg.inv(CLR_CVT_FROM_RGB_W[COLOR_YIQ]),
    COLOR_XYZ: np.linalg.inv(CLR_CVT_FROM_RGB_W[COLOR_XYZ])
}
CLR_CVT_TO_RGB_B = {
    COLOR_YCbCr: np.array([
        0, 128, 128
    ]),
    COLOR_YIQ: np.array([
        0, 0, 0
    ]),
    COLOR_XYZ: np.array([
        0, 0, 0
    ])
}


class Image:
    def __init__(self):
        self.im = None          # image data array
        self.mode = None        # color mode
        self._size = (0, 0)
        self.info = {}          # additional information
        self.readonly = 0

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    @property
    def size(self):
        return self._size

    def _new(self, im):
        new = Image()
        new.im = im
        new.mode = self.mode
        new._size = im.shape
        new.info = self.info.copy()
        return new

    def copy(self):
        return self._new(self.im)

    # TODO: YIQ, XYZ, HSI not fully tested yet
    def convert(self, mode=None):
        if mode == self.mode:
            return self.copy()
        if self.mode not in [COLOR_RGB, COLOR_YCbCr, COLOR_CMYK, COLOR_YIQ, COLOR_XYZ]:
            raise NotImplementedError('Current mode not supported yet.')
        if mode not in [COLOR_L, COLOR_RGB, COLOR_YCbCr, COLOR_CMYK, COLOR_YIQ, COLOR_XYZ, COLOR_HSI]:
            raise NotImplementedError('Target mode not supported yet.')

        if self.mode is not COLOR_RGB and mode is not COLOR_RGB:
            tmp = self.convert(COLOR_RGB)
            return tmp.convert(mode)

        new_im = np.zeros_like(self.im)

        if self.mode is COLOR_RGB:
            if mode is COLOR_L:
                new_im = np.dot(self.im, CLR_CVT_RGB2L.transpose() / 1000).astype(np.uint8)

            elif mode is COLOR_CMYK:
                new_im = (255 - self.im)
                k = np.min(new_im, axis=2, keepdims=True)
                new_im = np.uint8(new_im - k / (255 - k + 1e8))
                new_im = np.concatenate((new_im, k), axis=2)

            elif mode is COLOR_HSI:
                new_im = new_im.astype(np.float64)
                for i in range(self.height):
                    for j in range(self.width):
                        r, g, b = self.im[i, j, 0] / 255, self.im[i, j, 1] / 255, self.im[i, j, 2] / 255
                        new_im[i, j, 2] = np.mean(self.im[i, j, :])  # I channel
                        new_im[i, j, 1] = 1 - 3.0 / np.sum(self.im[i, j, :]) * np.min(self.im[i, j, :])  # S channel
                        new_im[i, j, 0] = np.arccos(
                            (2 * r - g - b) / 2 / (np.sqrt((r - g) ** 2 + (r - b) * (g - b))))  # H channel
                        if g < b:
                            new_im[i, j, 0] = 2 * np.pi - new_im[i, j, 0]

            else:
                new_im = np.uint8(np.dot(self.im, CLR_CVT_FROM_RGB_W[mode].transpose()) + CLR_CVT_FROM_RGB_B[mode])

        else:
            if self.mode is COLOR_CMYK:
                new_im = self.im / 255
                k = new_im[:, :, 3:]
                new_im = np.uint8((1.0 - new_im[:, :, :3]) * (1.0 - k) * 255)

            else:
                new_im = np.uint8(np.dot(self.im - CLR_CVT_TO_RGB_B[self.mode], CLR_CVT_TO_RGB_W[self.mode].transpose()))

        new = self._new(new_im)
        new.mode = mode

        return new

    def show(self, using='system'):
        if using not in ['system', 'matplotlib']:
            raise ValueError
        if using == 'system' and self.mode not in [COLOR_1, COLOR_L, COLOR_RGB, COLOR_YCbCr, COLOR_CMYK]:
            raise NotImplementedError

        if using == 'system':
            img_pil = PILImage.fromarray(self.im, self.mode)
            img_pil.show()
        else:
            plt.imshow(self.im)
            plt.show()

    def get_pixel(self, coord):
        x, y = coord

        if x < 0:
            x = self.width + x
        if y < 0:
            y = self.height + y
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError('Coordinate out of range!')

        return tuple(self.im[y, x])

    def crop(self, box=None):
        """
        Returns a rectangular region from this image. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.

        :param box: the crop rectangle, as a (left, upper, right, lower)-tuple
        :returns: a new Image object
        """
        if box is None:
            return self.copy()

        x0, y0, x1, y1 = map(int, map(round, box))
        if not (0 <= x0 < x1 < self.width and 0 <= y0 < y1 < self.height):
            raise ValueError

        im = self.im[y0:y1, x0:x1]

        return self._new(im)

    def histogram(self, bins=256):
        tmp = self.copy()
        tmp.convert(COLOR_L)

        hist = np.zeros(bins, dtype=int)
        for i in tmp.im.flatten():
            hist[i] += 1

        return hist

    def show_histogram(self, bins=256):
        if self.mode is COLOR_RGB:
            plt.figure("hist")
            arr_r = np.array(self.im[:, :, 0]).flatten()
            plt.hist(arr_r, bins=bins, facecolor='r', edgecolor='r', alpha=0.3)
            arr_g = np.array(self.im[:, :, 1]).flatten()
            plt.hist(arr_g, bins=bins, facecolor='g', edgecolor='g', alpha=0.3)
            arr_b = np.array(self.im[:, :, 2]).flatten()
            plt.hist(arr_b, bins=bins, facecolor='b', edgecolor='b', alpha=0.3)
            plt.show()
        elif self.mode is COLOR_L:
            plt.figure("hist")
            arr = np.array(self.im).flatten()
            plt.hist(arr, bins=bins, alpha=1)
            plt.show()
        else:
            raise NotImplementedError


def from_array(im, mode=COLOR_UNDEF):
    """
    Construct an Image object from numpy ndarray.

    :param im: ndarray with 3 dims (H, W, C)
    :param mode: color mode
    :return: an Image object
    """
    if not isinstance(im, np.ndarray):
        raise TypeError
    if not (im.ndim == 2 or (im.ndim == 3 and im.shape[2] in [1, 3])):
        return ValueError

    new = Image()
    new.im = im
    if mode is COLOR_UNDEF:
        new.mode = COLOR_RGB if im.ndim == 3 and im.shape[2] == 3 else COLOR_L
    else:
        new.mode = mode
    new._size = im.shape[:2]

    return new


def from_bmp(bmp, mode=COLOR_UNDEF):
    """
    Construct an Image object from Bmp object.

    :param bmp: a Bmp object
    :param mode: color mode
    :return: an Image object
    """
    if not isinstance(bmp, Bmp):
        raise TypeError

    im = bmp.to_array()

    return from_array(im, mode)


def from_bmp_file(filename, mode=COLOR_UNDEF):
    """
    Consturct an Image object from .bmp file.

    :param filename: name (path) of the .bmp file to read
    :param mode: color mode
    :return: an Image object
    """
    bmp = Bmp.from_file(filename)

    return from_bmp(bmp, mode)
