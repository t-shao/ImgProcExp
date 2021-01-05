import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as PILImage
from .bmp import Bmp

# color modes
COLOR_UNDEF = None
COLOR_L = 'L'
COLOR_RGB = 'RGB'
COLOR_YCbCr = 'YCbCr'
COLOR_YIQ = 'YIQ'
COLOR_HSI = 'HSI'
COLOR_XYZ = 'XYZ'

# color conversion matrix
CLR_CVT_RGB2L = np.array([299, 587, 114])
CLR_CVT_FROM_RGB = {
    COLOR_YCbCr: np.array([
        [.299, .587, .114],
        [-.1687, -.3313, .5],
        [.5, -.4187, .0813]
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
CLR_CVT_TO_RGB = {
    COLOR_YCbCr: np.linalg.inv(CLR_CVT_FROM_RGB[COLOR_YCbCr]),
    COLOR_YIQ: np.linalg.inv(CLR_CVT_FROM_RGB[COLOR_YIQ]),
    COLOR_XYZ: np.linalg.inv(CLR_CVT_FROM_RGB[COLOR_XYZ])
}


class Image:
    def __init__(self):
        self.im = None
        self.mode = None
        self._size = (0, 0)
        self.info = {}
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

    # TODO: further test needed
    def copy(self):
        return self._new(self.im)

    # TODO: not fully functional yet
    def convert(self, mode=None):
        if mode == self.mode:
            return
        if self.mode not in [COLOR_RGB, COLOR_YCbCr, COLOR_YIQ, COLOR_XYZ]:
            raise NotImplementedError('Current mode not supported yet.')
        if mode not in [COLOR_L, COLOR_RGB, COLOR_YCbCr, COLOR_YIQ, COLOR_XYZ]:
            raise NotImplementedError('Target mode not supported yet.')

        if self.mode is not COLOR_RGB and mode is not COLOR_RGB:
            self.convert(COLOR_RGB)
            self.convert(mode)
            return

        if self.mode is COLOR_RGB:
            if mode is COLOR_L:
                self.im = np.uint8(np.dot(self.im, CLR_CVT_RGB2L.transpose()) // 1000)

            else:
                self.im = np.uint8(np.dot(self.im, CLR_CVT_FROM_RGB[mode].transpose()))

        else:
            self.im = np.uint8(np.dot(self.im, CLR_CVT_TO_RGB[self.mode].transpose()))

        self.mode = mode

    def show(self):
        # plt.imshow(self.im)
        # plt.show()
        if self.mode not in [COLOR_L, COLOR_RGB, COLOR_YCbCr]:
            raise NotImplementedError

        img_pil = PILImage.fromarray(self.im, self.mode)
        img_pil.show()

    def get_pixel(self, coord):
        x, y = coord

        if x < 0:
            x = self.width + x
        if y < 0:
            y = self.height + y
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError('Coordinate out of range!')

        return tuple(self.im[y, x])

    # TODO: sanity check
    def crop(self, box=None):
        if box is None:
            return self.copy()

        x0, y0, x1, y1 = map(int, map(round, box))
        im = self.im[y0:y1, x0:x1]

        return self._new(im)

    def histogram(self):
        tmp = self.copy()
        tmp.convert(COLOR_L)

        hist = np.zeros(256, dtype=int)
        for i in tmp.im.flatten():
            hist[i] += 1

        return hist

    def show_histogram(self):
        if self.mode is COLOR_RGB:
            plt.figure("hist")
            arr_r = np.array(self.im[:, :, 0]).flatten()
            plt.hist(arr_r, bins=256, facecolor='r', edgecolor='r', alpha=0.3)
            arr_g = np.array(self.im[:, :, 1]).flatten()
            plt.hist(arr_g, bins=256, facecolor='g', edgecolor='g', alpha=0.3)
            arr_b = np.array(self.im[:, :, 2]).flatten()
            plt.hist(arr_b, bins=256, facecolor='b', edgecolor='b', alpha=0.3)
            plt.show()
        elif self.mode is COLOR_L:
            plt.figure("hist")
            arr = np.array(self.im).flatten()
            plt.hist(arr, bins=256, alpha=1)
            plt.show()
        else:
            raise NotImplementedError


def from_array(im, mode=COLOR_UNDEF):
    assert isinstance(im, np.ndarray) and im.ndim == 3 and im.shape[2] in [1, 3]

    new = Image()
    new.im = im
    if mode is COLOR_UNDEF:
        new.mode = COLOR_RGB if im.shape[2] == 3 else COLOR_L
    else:
        new.mode = mode
    new._size = im.shape[:2]

    return new


def from_bmp(bmp, mode=COLOR_UNDEF):
    assert isinstance(bmp, Bmp)

    im = bmp.to_array()

    return from_array(im, mode)


def from_bmp_file(filename, mode=COLOR_UNDEF):
    bmp = Bmp.from_file(filename)

    return from_bmp(bmp, mode)
