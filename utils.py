import torch
from torch import nn


torch.manual_seed(1)


def images_to_vectors(images, image_size):
    """
    Source: medium article
    """
    return images.view(images.size(0), image_size*image_size)


def vectors_to_images(vectors, image_size):
    """
    Source: medium article
    """
    return vectors.view(vectors.size(0), 1, image_size, image_size)
