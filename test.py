import numpy as np
from PIL import Image as PILImage
from matplotlib import pyplot as plt
from ImgProc import Bmp, Image, ImageOps, ImageFilter, Transform
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def test_1(filename):
    # read from the bmp file
    bmp = Bmp.Bmp.from_file(filename)

    # print file header and info header of the bmp file
    print(f"File: {filename}")
    bmp.print_file_header()
    bmp.print_info_header()

    # create an Image object and show the image
    im = bmp.to_array()
    img = Image.from_array(im)
    img.show('matplotlib')

    # convert color mode
    # to CMYK
    img_cmyk = img.convert(Image.COLOR_CMYK).im
    figure, ax = plt.subplots(1, 4)
    ax[0].imshow(img_cmyk[:, :, 0], cmap='gray')
    ax[0].set_title("C Channel")
    ax[1].imshow(img_cmyk[:, :, 1], cmap='gray')
    ax[1].set_title("M Channel")
    ax[2].imshow(img_cmyk[:, :, 2], cmap='gray')
    ax[2].set_title("Y Channel")
    ax[3].imshow(img_cmyk[:, :, 3], cmap='gray')
    ax[3].set_title("K Channel")
    plt.show()
    # to YCbCr
    img_ycbcr = img.convert(Image.COLOR_YCbCr).im
    figure, ax = plt.subplots(1, 3)
    ax[0].imshow(img_ycbcr[:, :, 0], cmap='gray')
    ax[0].set_title("Y Channel")
    ax[1].imshow(img_ycbcr[:, :, 1], cmap='gray')
    ax[1].set_title("Cb Channel")
    ax[2].imshow(img_ycbcr[:, :, 2], cmap='gray')
    ax[2].set_title("Cr Channel")
    plt.show()
    # to YIQ
    img_yiq = img.convert(Image.COLOR_YIQ).im
    figure, ax = plt.subplots(1, 3)
    ax[0].imshow(img_yiq[:, :, 0], cmap='gray')
    ax[0].set_title("Y Channel")
    ax[1].imshow(img_yiq[:, :, 1], cmap='gray')
    ax[1].set_title("I Channel")
    ax[2].imshow(img_yiq[:, :, 2], cmap='gray')
    ax[2].set_title("Q Channel")
    plt.show()
    # to XYZ
    img_xyz = img.convert(Image.COLOR_XYZ).im
    figure, ax = plt.subplots(1, 3)
    ax[0].imshow(img_xyz[:, :, 0], cmap='gray')
    ax[0].set_title("X Channel")
    ax[1].imshow(img_xyz[:, :, 1], cmap='gray')
    ax[1].set_title("Y Channel")
    ax[2].imshow(img_xyz[:, :, 2], cmap='gray')
    ax[2].set_title("Z Channel")
    plt.show()
    # to HSI
    '''
    img_hsi = img.convert(Image.COLOR_HSI).im
    figure, ax = plt.subplots(1, 3)
    ax[0].imshow(img_hsi[:, :, 0], cmap='gray')
    ax[0].set_title("H Channel")
    ax[1].imshow(img_hsi[:, :, 1], cmap='gray')
    ax[1].set_title("S Channel")
    ax[2].imshow(img_hsi[:, :, 2], cmap='gray')
    ax[2].set_title("I Channel")
    plt.show()
    '''

    # get pixel
    x = int(input('Enter x coordinate: '))
    y = int(input('Enter y coordinate: '))
    print(img.get_pixel((x, y)))


def test_2(img):
    Transform.test_dft(img)
    Transform.test_dct(img)


def test_3(img):
    # convert to monochrome
    img_l = img.convert(Image.COLOR_L)
    im_l = img_l.im

    # show and equalize / normalize the histogram
    img_l.show()
    img_l.show_histogram()
    # histogram equalization
    img_eq = ImageOps.equalize(img_l)
    img_eq.show()
    img_eq.show_histogram()
    # histogram normalization
    img_norm = ImageOps.normalize(img_l)
    img_norm.show()
    img_norm.show_histogram()

    # add noise and denoise
    # add gaussian / salt & pepper noise
    img_w_gn = ImageOps.add_gaussian_noise(img_l, var=0.01)
    im_w_gn = img_w_gn.im
    img_w_gn.show()
    img_w_spn = ImageOps.add_salt_pepper_noise(img_l)
    im_w_spn = img_w_spn.im
    img_w_spn.show()
    # apply median filter
    median_filter = ImageFilter.MedianFilter(3)
    im_d_gn_med = median_filter.filter(img_w_gn).im
    im_d_spn_med = median_filter.filter(img_w_spn).im
    # apply mean filter
    mean_filter = ImageFilter.MeanFilter(3)
    im_d_gn_mean = mean_filter.filter(img_w_gn).im
    im_d_spn_mean = mean_filter.filter(img_w_spn).im
    # apply dwt denoising
    im_d_gn_dwt = ImageOps.dwt_denoise(img_w_gn).im
    im_d_spn_dwt = ImageOps.dwt_denoise(img_w_spn).im
    # for gaussian noise
    print("Gaussian noise case: ")
    print("  with noise: ")
    print("    PSNR = {}".format(peak_signal_noise_ratio(im_l, im_w_gn)))
    print("    SSIM = {}".format(structural_similarity(im_l, im_w_gn)))
    print("  denoised by median filter: ")
    print("    PSNR = {}".format(peak_signal_noise_ratio(im_l, im_d_gn_med)))
    print("    SSIM = {}".format(structural_similarity(im_l, im_d_gn_med)))
    print("  denoised by mean filter: ")
    print("    PSNR = {}".format(peak_signal_noise_ratio(im_l, im_d_gn_mean)))
    print("    SSIM = {}".format(structural_similarity(im_l, im_d_gn_mean)))
    print("  denoised by DWT: ")
    print("    PSNR = {}".format(peak_signal_noise_ratio(im_l, im_d_gn_dwt)))
    print("    SSIM = {}".format(structural_similarity(im_l, im_d_gn_dwt)))
    figure, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(im_w_gn, cmap='gray')
    ax[0, 0].set_title("Guassian noise")
    ax[0, 1].imshow(im_d_gn_med, cmap='gray')
    ax[0, 1].set_title("Median filtered")
    ax[1, 0].imshow(im_d_gn_mean, cmap='gray')
    ax[1, 0].set_title("Mean filtered")
    ax[1, 1].imshow(im_d_gn_dwt, cmap='gray')
    ax[1, 1].set_title("DWT denoised")
    plt.show()
    # for salt & pepper noise
    print("Salt & pepper noise case: ")
    print("  with noise: ")
    print("    PSNR = {}".format(peak_signal_noise_ratio(im_l, im_w_spn)))
    print("    SSIM = {}".format(structural_similarity(im_l, im_w_spn)))
    print("  denoised by median filter: ")
    print("    PSNR = {}".format(peak_signal_noise_ratio(im_l, im_d_spn_med)))
    print("    SSIM = {}".format(structural_similarity(im_l, im_d_spn_med)))
    print("  denoised by mean filter: ")
    print("    PSNR = {}".format(peak_signal_noise_ratio(im_l, im_d_spn_mean)))
    print("    SSIM = {}".format(structural_similarity(im_l, im_d_spn_mean)))
    print("  denoised by DWT: ")
    print("    PSNR = {}".format(peak_signal_noise_ratio(im_l, im_d_spn_dwt)))
    print("    SSIM = {}".format(structural_similarity(im_l, im_d_spn_dwt)))
    figure, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(im_w_spn, cmap='gray')
    ax[0, 0].set_title("Salt&Pepper noise")
    ax[0, 1].imshow(im_d_spn_med, cmap='gray')
    ax[0, 1].set_title("Median filtered")
    ax[1, 0].imshow(im_d_spn_mean, cmap='gray')
    ax[1, 0].set_title("Mean filtered")
    ax[1, 1].imshow(im_d_spn_dwt, cmap='gray')
    ax[1, 1].set_title("DWT denoised")
    plt.show()

    # template operator
    # apply the Sobel operator
    sobel_h = ImageFilter.Sobel('h')
    sobel_v = ImageFilter.Sobel('v')
    img_sobel_h = sobel_h.filter(img_l)
    img_sobel_v = sobel_v.filter(img_l)
    img_sobel_h.show()
    img_sobel_v.show()
    # apply the Laplacian operator
    laplacian = ImageFilter.Laplacian()
    img_lap = laplacian.filter(img_l)
    img_lap.show()

    # binarization

    # binarize by thresholding
    img_bin = ImageOps.thresholing_binarize(img_l, 128)
    img_bin.show(using='matplotlib')
    # binarize by half-tone
    img_hf = ImageOps.half_toning_binarize(img_l)
    img_hf.show(using='matplotlib')
    # half-toning CMYK
    img_cmyk = img.convert(Image.COLOR_CMYK)
    img_cmyk_hf = ImageOps.half_tone(img_cmyk, size=3, angles=[15, 75, 0, 45], fill=1.0, sharpness=1.0)
    img_cmyk_hf.show()


if __name__ == '__main__':
    # read file
    filename = "data/lena.bmp"
    img = Image.from_bmp_file(filename)

    # Exp 1
    test_1(filename)

    # Exp 2
    test_2(img)

    # Exp 3
    test_3(img)
