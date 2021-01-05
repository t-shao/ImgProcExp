import numpy as np
from PIL import Image as PILImage
from ImgProc import Image, ImageOps

if __name__ == '__main__':
    filename = "data/lena.bmp"
    img = Image.from_bmp_file(filename)
    # print(img.get_pixel((0, 0)))
    img.convert(Image.COLOR_L)
    img.show()
    img_eq = ImageOps.equalize(img)
    img_eq.show()
    img_eq.show_histogram()
