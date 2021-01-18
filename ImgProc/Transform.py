import time

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

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


def get_magnitude_and_phase(im_freq, show_result=False):
    magnitude = np.absolute(im_freq)
    phase = np.arctan2(np.imag(im_freq), np.real(im_freq))

    if show_result is True:
        magnitude_normalized = (magnitude - np.amin(magnitude)) / (np.amax(magnitude) - np.amin(magnitude))
        phase_normalized = (phase - np.min(phase)) / (np.amax(phase) - np.amin(phase))

        figure, ax = plt.subplots(1, 2)
        ax[0].imshow(magnitude_normalized, cmap='gray')
        ax[0].set_title("Magnitude")
        ax[1].imshow(phase_normalized, cmap='gray')
        ax[1].set_title("Phase")
        plt.show()

    return magnitude, phase


def reconstruct_from_magnitude_or_phase(im_freq, show_result=False):
    magnitude, phase = get_magnitude_and_phase(im_freq)
    phase_ = np.exp(1j*phase)

    im_from_mag = np.real(idft2(magnitude)).round()
    im_from_pha = np.real(idft2(phase_)).round()

    if show_result is True:
        figure, ax = plt.subplots(1, 2)
        ax[0].imshow(im_from_mag, cmap='gray')
        ax[0].set_title("Reconstruction from Magnitude")
        ax[1].imshow(im_from_pha, cmap='gray')
        ax[1].set_title("Reconstruction from Phase")
        plt.show()

    return im_from_mag, im_from_pha


def dct_matrix(n):
    dct_mat = np.zeros((n, n))
    for j in range(n):
        dct_mat[0, j] = 1 / np.sqrt(n)
    for i in range(1, n):
        for j in range(n):
            dct_mat[i, j] = np.sqrt(2.0 / n) * np.cos(i * np.pi * (j + 0.5) / n)
    dct_mat = np.asmatrix(dct_mat)

    return dct_mat


def show_dct_basis(n=8):
    dct_mat = dct_matrix(n)
    basis = np.matmul(dct_mat.transpose().reshape((-1, 1), order='F'), dct_mat.reshape((1, -1)))
    basis_normalized = (basis - np.amin(basis)) / (np.amax(basis) - np.amin(basis))

    plt.imshow(basis_normalized, cmap='gray')
    plt.title("basis functions")
    plt.show()


def dct2(im, patch_size=8):
    n = patch_size
    h = im.shape[0] // n * n
    w = im.shape[1] // n * n

    im = im[0:h, 0:w]
    im_freq = np.zeros_like(im, dtype=np.complex_)
    dct_mat = dct_matrix(n)

    for i in range(0, h, n):
        for j in range(0, w, n):
            patch = im[i:i + n, j:j + n]
            im_freq[i:i + n, j:j + n] = np.matmul(np.matmul(dct_mat, patch), dct_mat.transpose())

    return im_freq


def idct2(im_freq, patch_size=8, top_k=1, show_result=False):
    n = patch_size
    h = im_freq.shape[0] // n * n
    w = im_freq.shape[1] // n * n

    im_freq = im_freq[0:h, 0:w]
    im = np.zeros_like(im_freq)
    dct_mat = dct_matrix(n)

    for i in range(0, h, n):
        for j in range(0, w, n):
            patch = im_freq[i:i + n, j:j + n]
            # retain top k coefficients
            patch_ = np.zeros_like(patch, dtype=np.complex_)
            row, col, s, dir = 0, 0, 0, 0  # dir: 0（right）, 1（left down）, 2（down）, 3（right up）
            # zig-zag scan
            while s < top_k:
                patch_[row, col] = patch[row, col]
                if dir == 0:
                    col += 1
                    if row == 0:
                        dir = 1
                    if row == n - 1:
                        dir = 3
                elif dir == 1:
                    col -= 1
                    row += 1
                    if row == n - 1:
                        dir = 0
                    elif col == 0:
                        dir = 2
                elif dir == 2:
                    row += 1
                    if col == 0:
                        dir = 3
                    if col == n - 1:
                        dir = 1
                elif dir == 3:
                    col += 1
                    row -= 1
                    if col == n - 1:
                        dir = 2
                    elif row == 0:
                        dir = 0
                else:
                    break
                s += 1

            im[i:i + n, j:j + n] = np.matmul(np.matmul(dct_mat.transpose(), patch_), dct_mat)

    im = np.real(im).round()

    if show_result is True:
        plt.imshow(im, cmap='gray')
        plt.title(f"Reconstruction from Top-{top_k} Coefficients")
        plt.show()

    return im


# TODO: dwt & idwt


def test_dft(image):
    # convert to monochrome
    img_l = image.convert(Image.COLOR_L)

    # do dft and show the magnitude and phase
    im = img_l.im
    im_freq = dft2(im, print_time=True)
    get_magnitude_and_phase(im_freq, show_result=True)

    # do idft and show the reconstruction from magnitude or phase respectively
    reconstruct_from_magnitude_or_phase(im_freq, show_result=True)


def test_dct(image):
    # convert to monochrome
    img_l = image.convert(Image.COLOR_L)

    # do dct and show dct basis
    im = img_l.im
    im_freq = dct2(im)
    show_dct_basis()

    # do idct with different # of coefficients and compute the PSNR
    im_gt = im[0:im_freq.shape[0], 0:im_freq.shape[1]]
    for k in [1, 2, 4, 6, 8, 10]:
        recon = idct2(im_freq, top_k=k, show_result=True)
        psnr = peak_signal_noise_ratio(im_gt, recon, data_range=255)
        print(f"top-{k}, psnr={psnr}")
