import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.tile_num = int(resolution/tile_size)
        self.output = []

    def draw(self):
        black_tile = np.zeros((self.tile_size, self.tile_size))
        white_tile = np.ones((self.tile_size, self.tile_size))
        bw_box = np.concatenate((black_tile, white_tile), axis=1)
        wb_box = np.concatenate((white_tile, black_tile), axis=1)
        box_2x2 = np.concatenate((bw_box, wb_box), axis=0)
        self.output = np.tile(box_2x2, (int(self.tile_num/2), int(self.tile_num/2)))
        output = self.output.copy()
        return output

    def show(self):
        img = self.output
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.show()


class Circle:

    def __init__(self, resolution, radius, position=(0, 0)):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = []

    def draw(self):
        x = np.linspace(0, self.resolution, self.resolution)
        y = np.linspace(0, self.resolution, self.resolution)
        xx, yy = np.meshgrid(x, y)

        circle = ((self.radius**2)-((xx-x[self.position[0]])**2 + (yy-self.position[1])**2))/(2*(self.resolution**2)) + 0.000001
        self.output = np.ceil(circle)
        output = self.output.copy()
        return output

    def show(self):
        img = self.output
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = []

    def draw(self):
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)

        xx, yy = np.meshgrid(x, y)
        r = xx
        g = yy
        b = np.flip(xx, axis=1)

        spectrum = np.zeros((self.resolution, self.resolution, 3))
        spectrum[:, :, 0] = r
        spectrum[:, :, 1] = g
        spectrum[:, :, 2] = b
        self.output = spectrum
        output = self.output.copy()
        return output

    def show(self):
        img = self.output
        plt.imshow(img)
        plt.show()


'''spectrum = Spectrum(1000)
spectrum.show()'''


'''circle = Circle(1000, 100, (500, 500))
circle.show()
print(circle.draw()[0])'''





