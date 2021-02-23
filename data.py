import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import torchvision


torch.manual_seed(1)


def load_mnist_data(classes, batch_size, image_size):
    # Load MNIST digits dataset and pre-process
    dataset = torchvision.datasets.MNIST('.', train=True, download=True,
                                    transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    
                                ]))
    
    # Only pick the chosen classes of digits
    if len(classes) == 0:
        raise Exception('classes list cannot be empty')
    idx = dataset.targets==classes[0]
    for c in classes[1:]:
        idx = torch.logical_or(idx, dataset.targets==c)
    dataset.data = dataset.data[idx]

    # Return dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def plot_data_batch(device, dataloader):
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


if __name__ == '__main__':
    # Plot some training images
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    dataloader = load_mnist_data(classes=[0,1], batch_size=128, image_size=28)
    plot_data_batch(device, dataloader)