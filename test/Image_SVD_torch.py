''' Written by Omolewa '''

import torch
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [16, 8]


def plot_image(image, image_title, plot_type = 'image', singular_values = None):
    """

    :param image:
    :param image_title:
    :param plot_type:
    :param singular_values:
    :return:
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
    image = torch.mean(image, -1) # convert RGB to grayscale
    U, S, VT = torch.linalg.svd(image, full_matrices=False)
    S = torch.diag(S)

    return U, S, VT


def plot_decompose(image, r, plot_type=None):
    U, S, VT = decompose(image)
    ImageApprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    img_title = f'r = {r}'
    if plot_type:
        plot_image(ImageApprox, r)
    else:
        plot_image(image=ImageApprox, plot_type='singular_values', image_title=img_title, singular_values=S)


if __name__ == '__main__':
    A = torch.from_numpy(imread('../test/michelle.jpeg'))
    A = torch.Tensor.float(A)

    for r in [20, 50, 100, 200]:
        plot_decompose(A, r, 'image')


