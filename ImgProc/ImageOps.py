import random

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import rotate

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


def normalize(image, min_max=(0, 255)):
    if not isinstance(image, Image.Image):
        raise TypeError

    h, w = image.size
    im = image.im
    pix_min, pix_max = np.amin(im), np.amax(im)
    tar_min, tar_max = min_max[0], min_max[1]

    new_im = np.uint8((tar_max - tar_min) / (pix_max - pix_min) * (im - pix_min) + tar_min)

    new = image.copy()
    new.im = new_im

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


def dwt_denoise(image):
    if not isinstance(image, Image.Image):
        raise TypeError

    new = image.convert(Image.COLOR_L)

    coeffs = pywt.wavedec2(new.im, 'db1', level=2)
    ths = [23.38, 10.12]

    for i in range(1, len(coeffs)):
        coeffs[i] = tuple([pywt.threshold(v, ths[i - 1], 'hard') for v in coeffs[i]])

    new.im = np.uint8(pywt.waverec2(coeffs, 'db1'))

    return new


def _resize(a, shape):
    """
    if array a is larger than shape, crop a; if a is smaller than shape, pad a with zeros
        Args:
            a (numpy array): 2D array to resize
            shape: desired shape of the return
        Returns:
            numpy array: array a resized according to shape
    """
    if a.shape[0] < shape[0]:
        a = np.pad(a, ((0, shape[0] - a.shape[0]), (0, 0)), mode="constant")
    if a.shape[1] < shape[1]:
        a = np.pad(a, ((0, 0), (0, shape[1] - a.shape[1])), mode="constant")
    if a.shape[0] > shape[0]:
        a = a[0:shape[0], :]
    if a.shape[1] > shape[1]:
        a = a[:, 0:shape[1]]
    return a


def _crop_center(img, new_shape):
    """
    Crop an image equally on each size to create the new_shape
        Args:
            img (numpy array): 2D array to crop
            new_shape: desired shape of the return
        Returns:
            numpy array: array cropped according to shape
    """
    ul = ((img.shape[0] - new_shape[0]) // 2, (img.shape[1] - new_shape[1]) // 2)
    br = (ul[0] + new_shape[0], ul[1] + new_shape[1])
    return img[ul[0]:br[0], ul[1]:br[1]]


def _gauss_kernel(size, sigma=None, size_y=None, sigma_y=None):
    """
    Generates a 2D Gaussian kernel as a numpy array
        Args:
            size (int): 1/2 the width of the kernel; total width := 2*size+1
            sigma (float): spread of the gaussian in the width direction
            size_y (int): 1/2 the height of the kernel; defaults to size
            sigma_y (float): spread of the gaussian in the height direction; defaults to sigma
        Returns:
            numpy array: normalized 2D gaussian array
    """
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    if not sigma:
        sigma = 0.5 * size + .1
    if not sigma_y:
        sigma_y = sigma
    x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]
    g = np.exp(-0.5 * (x ** 2 / sigma ** 2 + y ** 2 / sigma_y ** 2))
    return g / g.sum()


def thresholing_binarize(image, threshold):
    if not isinstance(image, Image.Image):
        raise TypeError

    im = image.im
    new = image.copy()
    new_im = np.zeros_like(im, dtype=bool)
    new.mode = Image.COLOR_1

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            new_im[i, j] = True if im[i, j] >= threshold else False

    new.im = new_im

    return new


def half_toning_binarize(image):
    if not isinstance(image, Image.Image):
        raise TypeError

    m = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 1, 0], [0, 0, 0], [0, 0, 1]],
                  [[1, 1, 0], [0, 0, 0], [0, 0, 1]],
                  [[1, 1, 0], [0, 0, 0], [1, 0, 1]],
                  [[1, 1, 1], [0, 0, 0], [1, 0, 1]],
                  [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
                  [[1, 1, 1], [0, 0, 1], [1, 1, 1]],
                  [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                  [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]) * 256

    im = image.convert(Image.COLOR_L).im
    h, w = im.shape
    new_im = np.zeros((3 * h, 3 * w), dtype=np.bool_)
    step = int(np.ceil(256 / 10))
    img_ten = np.fix(im / step).astype(np.uint8)

    for i in range(h):
        for j in range(w):
            gray_level = img_ten[i, j]
            new_im[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)] = m[gray_level]

    new = Image.from_array(new_im)
    new.mode = Image.COLOR_1

    return new


def half_tone(image, size, angles, fill, sharpness):
    if not isinstance(image, Image.Image):
        raise TypeError
    if image.mode not in [Image.COLOR_L, Image.COLOR_CMYK]:
        raise ValueError

    image_hf = image.copy()
    im = image.im / 255

    if image.mode is Image.COLOR_L:
        im = np.expand_dims(im, 2)
    im_hf = np.zeros_like(im, dtype=np.uint8)

    for i, (channel, angle) in enumerate(zip(np.rollaxis(im, 2), angles)):
        # total width of the kernel
        s = 2 * size + 1

        # rotate the image to eliminate overlap between the channels
        rotated = rotate(channel, angle, reshape=True, prefilter=False, order=1)

        # apply a gaussian filter to average over a the region of the kernel
        averaged = gaussian_filter(rotated, size)

        # find the central value of the filtered image; this is the average intensity in the region
        halftone_weights = averaged[size::s, size::s]

        # tile the weight image with the average intensity value
        halftone_weights = np.repeat(np.repeat(halftone_weights, s, 0), s, 1)
        halftone_weights = _resize(halftone_weights, rotated.shape)

        # TODO: consider using sigma to scale with magnitude
        # create a 2D gaussian kernel that will be the "dot"; normalize it to be 1.0 in the center
        kernel = _gauss_kernel(size, sigma=fill * size)

        # Apply the sharpness multiplier and clip the kernel to 1.0
        kernel *= sharpness / np.max(kernel)
        kernel = np.clip(kernel, 0.0, 1.0)

        # tile the kernel across the image
        num_kernels = np.array(rotated.shape) // s + 1
        tiled_kernel = np.tile(kernel, num_kernels)
        tiled_kernel = _resize(tiled_kernel, rotated.shape)

        # multiply the kernel image with the weights to generate the halftone image
        halftone = tiled_kernel * halftone_weights

        # rotate the image back to zero
        halftone = rotate(halftone, -angle, prefilter=False, order=1)

        # crop the image to the original size
        halftone = _crop_center(halftone, channel.shape)

        # add this chanel to the full cmyk image
        im_hf[:, :, i] = (halftone * 255).astype(np.uint8)

    if image.mode is Image.COLOR_L:
        im_hf = np.squeeze(im_hf, 2)
    image_hf.im = im_hf

    return image_hf
