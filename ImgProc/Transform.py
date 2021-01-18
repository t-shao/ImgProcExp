import time

import matplotlib.pyplot as plt
import numpy as np

from . import Image


def dft_matrix(n):
    dft_mat = np.zeros((n, n), dtype=np.complex_)
    for i in range(n):
        for j in range(n):
            dft_mat[i, j] = np.exp(-2j * np.pi * (i * j / n))
    dft_mat = np.asmatrix(dft_mat)

    return dft_mat


def dft2(im, patch_size=8, print_time=False):
    n = patch_size
    h = im.shape[0] // n * n
    w = im.shape[1] // n * n

    im = im[0:h, 0:w]
    im_freq = np.zeros_like(im, dtype=np.complex_)
    dft_mat = dft_matrix(n)

    start = time.time()
    for i in range(0, h, n):
        for j in range(0, w, n):
            patch = im[i:i + n, j:j + n]
            im_freq[i:i + n, j:j + n] = (1 / n) * np.matmul(np.matmul(dft_mat, patch), dft_mat.H)
    end = time.time()

    if print_time is True:
        print(f"DFT operation costs {end - start} s")

    return im_freq


def idft2(im_freq, patch_size=8):
    n = patch_size
    h = im_freq.shape[0] // n * n
    w = im_freq.shape[1] // n * n

    im_freq = im_freq[0:h, 0:w]
    im = np.zeros_like(im_freq, dtype=np.complex_)
    dft_mat = dft_matrix(n)

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = im_freq[i:i + n, j:j + n]
            im[i:i + n, j:j + n] = (1 / n) * np.matmul(np.matmul(dft_mat.H, patch), dft_mat)

    return im


def get_magnitude_and_phase(im_freq):
    magnitude = np.absolute(im_freq)
    phase = np.arctan2(np.imag(im_freq), np.real(im_freq))

    return magnitude, phase


def show_magnitude_and_phase(im_freq):
    magnitude, phase = get_magnitude_and_phase(im_freq)

    magnitude_normalized = (magnitude - np.amin(magnitude)) / (np.amax(magnitude) - np.amin(magnitude))
    phase_normalized = (phase - np.min(phase)) / (np.amax(phase) - np.amin(phase))

    figure, ax = plt.subplots(1, 2)
    ax[0].imshow(magnitude_normalized, cmap='gray')
    ax[0].set_title("Magnitude")
    ax[1].imshow(phase_normalized, cmap='gray')
    ax[1].set_title("Phase")
    plt.show()


def reconstruct_from_magnitude_or_phase(im_freq):
    magnitude, phase = get_magnitude_and_phase(im_freq)
    phase_ = np.exp(1j*phase)

    im_from_mag = np.real(idft2(magnitude))
    im_from_pha = np.real(idft2(phase_))

    figure, ax = plt.subplots(1, 2)
    ax[0].imshow(im_from_mag, cmap='gray')
    ax[0].set_title("Reconstruction from Magnitude")
    ax[1].imshow(im_from_pha, cmap='gray')
    ax[1].set_title("Reconstruction from Phase")
    plt.show()


# TODO: dct2 & idct2

# TODO: dwt & idwt


def test_dft(image):
    # convert to monochrome and show
    image.convert(Image.COLOR_L)
    image.show()

    # do dft and show the magnitude and phase
    im = image.im
    im_freq = dft2(im, print_time=True)
    show_magnitude_and_phase(im_freq)

    # do idft and show the reconstruction from magnitude or phase respectively
    reconstruct_from_magnitude_or_phase(im_freq)
