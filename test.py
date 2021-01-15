import numpy as np
from PIL import Image as PILImage
from matplotlib import pyplot as plt
from ImgProc import Image, ImageOps, ImageFilter, Transform

if __name__ == '__main__':
    '''read file'''
    filename = "data/img_100.bmp"
    img = Image.from_bmp_file(filename)

    '''show the image'''
    # plt.imshow(img.im)
    # plt.show()
    # img.show()

    ''' get pixel '''
    # print(img.get_pixel((0, 0)))

    '''convert to monochrome'''
    img.convert(Image.COLOR_L)
    img.show()

    '''binarize by thresholding'''
    img_bin = ImageOps.thresholing_binarize(img, 200)
    plt.imshow(img_bin.im)
    plt.show()

    '''crop into a patch'''
    # img_se = img.crop((0, 0, 32, 32))
    # img_se.show()

    '''show and equalize the histogram'''
    # img.show_histogram()
    # img_eq = ImageOps.equalize(img)
    # img_eq.show()
    # img_eq.show_histogram()

    '''add gaussian noise'''
    # img_w_gn = ImageOps.add_gaussian_noise(img)
    # img_w_gn.show()

    '''add salt & pepper noise'''
    # img_w_spn = ImageOps.add_salt_pepper_noise(img)
    # img_w_spn.show()

    '''apply the median filter'''
    # median_filter = ImageFilter.MedianFilter(3)
    # img_dn_med = median_filter.filter(img_w_spn)
    # img_dn_med.show()

    '''apply the mean filter'''
    # mean_filter = ImageFilter.MeanFilter(3)
    # img_dn_mean = mean_filter.filter(img_w_spn)
    # img_dn_mean.show()

    '''apply the sobel operator'''
    # sobel_v = ImageFilter.Sobel('v')
    # img_sobel_v = sobel_v.filter(img)
    # img_sobel_v.show()

    '''apply the robert operator'''
    # robert_45 = ImageFilter.Robert(45)
    # img_rob = robert_45.filter(img)
    # img_rob.show()

    '''apply the laplacian operator'''
    # laplacian = ImageFilter.Laplacian()
    # img_lap = laplacian.filter(img)
    # img_lap.show()

    '''dft'''
    # img_freq = Transform.dft2(img_se)
    # plt.imshow(abs(img_freq))
    # plt.show()
