import numpy as np
import random
from . import Image


def equalize(image):
    if not isinstance(image, Image.Image):
        raise TypeError
    if image.mode is not Image.COLOR_L:
        raise NotImplementedError
    h, w = image.size
    hist = image.histogram()

    new = image.copy()
    im = new.im

    # calculate cdf
    hist_cumsum = np.cumsum(hist)
    const_a = 256 / (h * w)
    hists_cdf = (const_a * hist_cumsum).astype(np.uint8)

    # mapping
    new.im = hists_cdf[im]

    return new


def add_gaussian_noise(image, mean=0, var=0.01):
    if not isinstance(image, Image.Image):
        raise TypeError

    new = image.copy()
    im = np.array(new.im / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, im.shape)
    im = np.uint8(np.clip(im + noise, 0, 1) * 255)
    new.im = im

    return new


def add_salt_pepper_noise(image, prob=0.01):
    if not isinstance(image, Image.Image):
        raise TypeError

    thres = 1 - prob
    new = image.copy()
    im = new.im
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            rand = random.random()
            if rand < prob:
                im[i][j] = 0
            elif rand > thres:
                im[i][j] = 255
    new.im = im

    return new

# TODO: binarization (thresholding and half-tone)
