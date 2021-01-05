import numpy as np


class BmpFileHeader:
    def __init__(self):
        self._bfType = int_to_bytes(0, 2)  # 0x4d42
        self._bfSize = int_to_bytes(0, 4)  # file size
        self._bfReserved1 = int_to_bytes(0, 2)
        self._bfReserved2 = int_to_bytes(0, 2)
        self._bfOffBits = int_to_bytes(0, 4)  # header info offset


class BmpInfoHeader:
    def __init__(self):
        self._biSize = int_to_bytes(0, 4)  # header size
        self._biWidth = int_to_bytes(0, 4)
        self._biHeight = int_to_bytes(0, 4)
        self._biPlanes = int_to_bytes(0, 2)  # default 1
        self._biBitCount = int_to_bytes(0, 2)  # one pixel occupy how many bits
        self._biCompression = int_to_bytes(0, 4)
        self._biSizeImage = int_to_bytes(0, 4)
        self._biXPelsPerMeter = int_to_bytes(0, 4)
        self._biYPelsPerMeter = int_to_bytes(0, 4)
        self._biClrUsed = int_to_bytes(0, 4)
        self._biClrImportant = int_to_bytes(0, 4)


class Bmp(BmpFileHeader, BmpInfoHeader):
    def __init__(self):
        BmpFileHeader.__init__(self)
        BmpInfoHeader.__init__(self)
        # size of data in bytes
        self.__dataSize = 0
        # pixel data in bytes
        self.data = []

    @property
    def width(self):
        return bytes_to_int(self._biWidth)

    @property
    def height(self):
        return bytes_to_int(self._biHeight)

    @property
    def bit_per_pixel(self):
        return bytes_to_int(self._biBitCount)

    # deprecated
    @property
    def byte_per_pixel(self):
        return self.bit_per_pixel // 8

    @property
    def channels(self):
        if self.bit_per_pixel == 8:
            return 1
        elif self.bit_per_pixel == 24:
            return 3
        else:
            raise NotImplementedError("Currently only 8/24 bit per pixel are supported.")

    @staticmethod
    def from_file(filename):
        new = Bmp()
        file = open(filename, 'rb')

        new._bfType = file.read(2)
        new._bfSize = file.read(4)
        new._bfReserved1 = file.read(2)
        new._bfReserved2 = file.read(2)
        new._bfOffBits = file.read(4)

        new._biSize = file.read(4)
        new._biWidth = file.read(4)
        new._biHeight = file.read(4)
        new._biPlanes = file.read(2)
        new._biBitCount = file.read(2)
        new._biCompression = file.read(4)
        new._biSizeImage = file.read(4)
        new._biXPelsPerMeter = file.read(4)
        new._biYPelsPerMeter = file.read(4)
        new._biClrUsed = file.read(4)
        new._biClrImportant = file.read(4)
        # to be improved
        byte_per_row = (new.width * new.byte_per_pixel + 3) // 4 * 4
        new.__dataSize = byte_per_row * new.height
        # to be improved
        for i in range(new.__dataSize):
            new.data.append(file.read(1))

        file.close()
        return new

    def save(self, filename):
        file = open(filename, 'wb')

        file.write(self._bfType)
        file.write(self._bfSize)
        file.write(self._bfReserved1)
        file.write(self._bfReserved2)
        file.write(self._bfOffBits)

        file.write(self._biSize)
        file.write(self._biWidth)
        file.write(self._biHeight)
        file.write(self._biPlanes)
        file.write(self._biBitCount)
        file.write(self._biCompression)
        file.write(self._biSizeImage)
        file.write(self._biXPelsPerMeter)
        file.write(self._biYPelsPerMeter)
        file.write(self._biClrUsed)
        file.write(self._biClrImportant)

        # color table omitted
        # to be improved
        for byte in self.data:
            file.write(byte)

        file.close()

    def get_pixel(self, x_coord, y_coord):
        assert 0 <= x_coord < self.width and 0 <= y_coord < self.height

        pixel_data = list()
        byte_per_row = (self.width * self.byte_per_pixel + 3) // 4 * 4
        idx = (self.height - y_coord - 1) * byte_per_row + x_coord * self.byte_per_pixel
        for i in range(self.byte_per_pixel):
            pixel_data.append(self.data[idx + i])

        if self.bit_per_pixel in [8, 24]:
            pixel_data = list(map(bytes_to_int, pixel_data))
            pixel_data = tuple(reversed(pixel_data))

        return pixel_data

    def to_array(self):
        array = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)

        for i in range(self.height):
            for j in range(self.width):
                array[i, j] = self.get_pixel(j, i)

        return array


def int_to_bytes(number, length, byteorder='little'):
    return number.to_bytes(length, byteorder)


def bytes_to_int(bytes, byteorder='little'):
    return int.from_bytes(bytes, byteorder)


if __name__ == "__main__":
    bmp = Bmp.from_file('data/zebra.bmp')
    bmp.save('processed.bmp')
    #x, y = map(int, input("enter coord: ").split())
    #pix = bmp.get_pixel(x, y)
    #print(pix)
    img = bmp.to_array()
    print(img.shape, img.dtype)
    print(img)

