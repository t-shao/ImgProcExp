import numpy as np


class BmpFileHeader:
    def __init__(self):
        self._bfType = int_to_bytes(0, 2)  # file type: 0x4d42
        self._bfSize = int_to_bytes(0, 4)  # file size
        self._bfReserved1 = int_to_bytes(0, 2)
        self._bfReserved2 = int_to_bytes(0, 2)
        self._bfOffBits = int_to_bytes(0, 4)  # offset


class BmpInfoHeader:
    def __init__(self):
        self._biSize = int_to_bytes(0, 4)  # header size
        self._biWidth = int_to_bytes(0, 4)
        self._biHeight = int_to_bytes(0, 4)
        self._biPlanes = int_to_bytes(0, 2)
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

    @property
    def byte_per_pixel(self):
        """
        (deprecated) do not support circumstances of 1/2/4 bit per pixel
        """
        return self.bit_per_pixel // 8

    @property
    def channels(self):
        """
        (not fully implemented yet)

        :return: number of (color) channels
        """
        if self.bit_per_pixel == 8:
            return 1
        elif self.bit_per_pixel == 24:
            return 3
        else:
            raise NotImplementedError("Currently only 8/24 bit per pixel are supported.")

    def print_file_header(self):
        print("File header:")
        print("bfType: {}".format(self._bfType))
        print("bfSize: {}".format(bytes_to_int(self._bfSize)))
        print("bfReserved1: {}".format(bytes_to_int(self._bfReserved1)))
        print("bfReserved2: {}".format(bytes_to_int(self._bfReserved2)))
        print("bfOffBits: {}".format(bytes_to_int(self._bfOffBits)))

    def print_info_header(self):
        print("Info header:")
        print("biSize: {}".format(bytes_to_int(self._biSize)))
        print("biWidth: {}".format(bytes_to_int(self._biWidth)))
        print("biHeight: {}".format(bytes_to_int(self._biHeight)))
        print("biPlanes: {}".format(bytes_to_int(self._biPlanes)))
        print("biBitCount: {}".format(bytes_to_int(self._biBitCount)))
        print("biCompression: {}".format(bytes_to_int(self._biCompression)))
        print("biSizeImage: {}".format(bytes_to_int(self._biSizeImage)))
        print("biXPelsPerMeter: {}".format(bytes_to_int(self._biXPelsPerMeter)))
        print("biYPelsPerMeter: {}".format(bytes_to_int(self._biYPelsPerMeter)))
        print("biClrUsed: {}".format(bytes_to_int(self._biClrUsed)))
        print("biClrImportant: {}".format(bytes_to_int(self._biClrImportant)))

    @staticmethod
    def from_file(filename):
        """
        create a Bmp object from .bmp file.
        (not fully implemented yet)

        :param filename: name (path) of the .bmp file to read
        :return: a Bmp object
        """
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

        # color table is not considered for now
        # TODO: capability to read the color table

        # align to 4 bytes for each row
        byte_per_row = (new.width * new.byte_per_pixel + 3) // 4 * 4
        new.__dataSize = byte_per_row * new.height
        # might can be improved
        for i in range(new.__dataSize):
            new.data.append(file.read(1))

        file.close()
        return new

    def save(self, filename):
        """
        save the Bmp object into a .bmp file.

        :param filename: name (path) of the .bmp file to save
        """
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

        # color table omitted for now

        for byte in self.data:
            file.write(byte)

        file.close()

    def get_pixel(self, coord):
        """
        get the pixel value of the specific position given by coordinate.

        :param coord: coordinate (x, y)
        :return: pixel value in tuple
        """
        x, y = coord
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError('Coordinate out of range!')

        pixel_data = list()
        byte_per_row = (self.width * self.byte_per_pixel + 3) // 4 * 4
        idx = (self.height - y - 1) * byte_per_row + x * self.byte_per_pixel
        for i in range(self.byte_per_pixel):
            pixel_data.append(self.data[idx + i])

        if self.bit_per_pixel in [8, 24]:
            pixel_data = list(map(bytes_to_int, pixel_data))
            pixel_data = tuple(reversed(pixel_data))

        return pixel_data

    def to_array(self):
        """
        construct a numpy ndarray with the data of the Bmp object.

        :return: ndarray with 3 dims (H, W, C)
        """
        array = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)

        for i in range(self.height):
            for j in range(self.width):
                array[i, j] = self.get_pixel((j, i))

        return array


def int_to_bytes(number, length, byteorder='little'):
    return number.to_bytes(length, byteorder)


def bytes_to_int(bytes, byteorder='little'):
    return int.from_bytes(bytes, byteorder)


if __name__ == "__main__":
    bmp = Bmp.from_file('../data/zebra.bmp')
    bmp.save('processed.bmp')
    #x, y = map(int, input("enter coord: ").split())
    #pix = bmp.get_pixel(x, y)
    #print(pix)
    img = bmp.to_array()
    print(img.shape, img.dtype)
    print(img)

