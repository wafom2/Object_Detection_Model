
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['figure.figsize'] = [16, 8]

A = imread('../test/michelle.jpeg')


def plot_image(image, image_title, plot_type='image', singular_values=None, singular_val_cumm = None):
    """

    :param image:
    :param image_title:
    :param plot_type:
    :param singular_values:
    :param singular_val_cumm:
    :return:jpeg -> low rank image
    """

    if plot_type == 'image':
        img = plt.imshow(256 - image)
        img.set_cmap('gray')
        plt.axis('off')
    elif plot_type == 'singular_values':
        plt.figure(1)
        plt.semilogy(singular_values)

        plt.figure(2)
        plt.plot(np.cumsum(singular_values / np.sum(singular_values)))

    plt.title(f'{image_title}')
    plt.show()


def decompose(image):
    """

    :param image: jpeg -> image to decompose
    :return: tuple -> tuple of matrices
    """
    image = np.mean(image, -1) # convert RGB to grayscale
    U, S, VT = np.linalg.svd(image, full_matrices=False) # performing singular value decomposition
    S = np.diag(S)

    return U, S, VT


def plot_decompose(image, r):
    U, S, VT = decompose(image)
    ImageApprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    img_title = f'r = {r}'
    plot_image(ImageApprox, img_title)


if __name__ == '__main__':
    plot_decompose(A, 200)
