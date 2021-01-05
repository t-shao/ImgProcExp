import numpy as np
from PIL import Image as PILImage
from ImgProc import Image, ImageOps

if __name__ == '__main__':
    filename = "data/img_100.bmp"
    img = Image.from_bmp_file(filename)
    img.show()
    # print(img.get_pixel((0, 0)))
    # img.convert(Image.COLOR_L)
    # img.show()
    # img.show_histogram()
    # img_eq = ImageOps.equalize(img)
    # img_eq.show()
    # img_eq.show_histogram()
    img_w_gn = ImageOps.add_gaussian_noise(img)
    img_w_gn.show()
    img_w_spn = ImageOps.add_salt_pepper_noise(img)
    img_w_spn.show()
