import numpy as np
from matplotlib import pyplot as plt
from bmp import Bmp

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

    @staticmethod
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

    @staticmethod
    def from_bmp(bmp, mode=COLOR_UNDEF):
        assert isinstance(bmp, Bmp)

        im = bmp.to_array()

        return Image.from_array(im, mode)

    @staticmethod
    def from_bmp_file(filename, mode=COLOR_UNDEF):
        bmp = Bmp.from_file(filename)

        return Image.from_bmp(bmp, mode)

    def convert(self, mode=None):
        if self.mode not in [COLOR_RGB, COLOR_YCbCr, COLOR_YIQ, COLOR_XYZ]:
            raise NotImplementedError('Current mode not supported yet.')
        if mode not in [COLOR_L, COLOR_RGB, COLOR_YCbCr, COLOR_YIQ, COLOR_XYZ]:
            raise NotImplementedError('Target mode not supported yet.')

        if mode == self.mode:
            return
        elif self.mode is not COLOR_RGB and mode is not COLOR_RGB:
            self.convert(COLOR_RGB)
            self.convert(mode)
            return

        if self.mode is COLOR_RGB:
            if mode is COLOR_L:
                self.convert(COLOR_YCbCr)
                self.im = np.uint8(np.dot(self.im, CLR_CVT_RGB2L.transpose()) // 1000)

            else:
                self.im = np.uint8(np.dot(self.im, CLR_CVT_FROM_RGB[mode].transpose()))

        else:
            self.im = np.uint8(np.dot(self.im, CLR_CVT_TO_RGB[self.mode].transpose()))

        self.mode = mode

    def show(self):
        plt.imshow(self.im)

