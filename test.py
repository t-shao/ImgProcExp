import numpy as np
from PIL import Image as PILImage
from matplotlib import pyplot as plt
from ImgProc import Image, ImageOps, ImageFilter, Transform

if __name__ == '__main__':
    filename = "data/img_100.bmp"
    img = Image.from_bmp_file(filename)
    # plt.imshow(img.im)
    # plt.show()
    # img.show()
    # print(img.get_pixel((0, 0)))
    img.convert(Image.COLOR_L)
    # img_se = img.crop((0, 0, 32, 32))
    # img_se.show()
    # img.show()
    # img.show_histogram()
    # img_eq = ImageOps.equalize(img)
    # img_eq.show()
    # img_eq.show_histogram()
    # img_w_gn = ImageOps.add_gaussian_noise(img)
    # img_w_gn.show()
    img_w_spn = ImageOps.add_salt_pepper_noise(img)
    img_w_spn.show()
    median_filter = ImageFilter.MedianFilter(3)
    mean_filter = ImageFilter.MeanFilter(3)
    img_dn_med = median_filter.filter(img_w_spn)
    img_dn_med.show()
    img_dn_mean = mean_filter.filter(img_w_spn)
    img_dn_mean.show()
    # img_freq = Transform.dft2(img_se)
    # plt.imshow(abs(img_freq))
    # plt.show()
