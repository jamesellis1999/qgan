"""
Code for this project takes inspiration from the following tutorials:

- DCGAN PyTorch (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- Medium article (https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f)
- Youtube tutorial (https://www.youtube.com/watch?v=OljTVUVzPpM&ab_channel=AladdinPersson)
"""


from data import load_mnist_data, plot_data_batch
from utils import images_to_vectors, vectors_to_images
from generator import Generator_Quantum
from discriminator import Discriminator_FCNN, DiscriminatorNet


import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.tensorboard import SummaryWriter


# Set seeds
torch.manual_seed(1)

# Set hyperparameters
batch_size = 128  # Batch size during training
fixed_noise_batch_size = 64  # Batch size for latent vector
image_size = 8  # Spatial size of training images. All images will be resized to this size using a transformer.
nz = 3 # Size of z latent vector (i.e. size of generator input)
num_epochs = 3  # Number of training epochs
lr = 3e-4  # Learning rate for optimizers
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
ngpu = 1  # Number of GPUs available. Use 0 for CPU mode

n_qubits = 3
q_depth = 3
q_delta = 1 # Spread of random distribution for initialising paramaterised quantum gates

def main():
    # Load data
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    dataloader = load_mnist_data(classes=[0,1], batch_size=batch_size, image_size=image_size)

    # Plot some training images
    #plot_data_batch(device, dataloader)

    # Create the generator
    netG = Generator_Quantum(n_qubits, q_depth, q_delta).to(device)

    # Create the Discriminator
    netD = Discriminator_FCNN(image_size*image_size, ngpu).to(device)

    train(device, netG, netD, dataloader)


def train(device, netG, netD, dataloader):
    """
    Train GAN
    """
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    
    # Set to 2 at the minute for 2 active qubits
    fixed_noise = torch.rand(fixed_noise_batch_size, nz, device=device) * np.pi

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


    # Outputs
    writer_fake = SummaryWriter(f"test/GAN_MNIST/fake_q")
    writer_real = SummaryWriter(f"test/GAN_MNIST/real_q")
    writer_loss = SummaryWriter(f"test/GAN_MNIST/loss_q")
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):  # Iterate over all epochs
        for i, data in enumerate(dataloader, 0):  # Iterate over all batches

            """
            (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            """
           
            ## Train with all-real batch
            netD.zero_grad()
            real_cpu = images_to_vectors(data[0].to(device), image_size)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)  # log(D(x))
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.rand(b_size, nz, device=device) * np.pi
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)  # log(1-D(G(z)))
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            optimizerD.step()

            """
            (2) Update G network: maximize log(D(G(z)))
            """

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)  # log(D(G(z)))
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            """
            (3) Prepare data for output
            """

            # Output training stats
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Check how the generator is doing by saving G's output on fixed_noise
            """
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = vectors_to_images(netG(fixed_noise).detach().cpu(), image_size)
                img_list.append(vutils.make_grid(fake, padding=2, normalize=False))
            """

            # Tensorboard - print once at the beginning of each epoch
            if i == 0:
                # Save errors
                G_losses.append(errG.item())
                writer_loss.add_scalar('G training loss', errG.item(), global_step=epoch)
                D_losses.append(errD.item())
                writer_loss.add_scalar('D training loss', errD.item(), global_step=epoch)

                # Save images
                with torch.no_grad():
                    fake = vectors_to_images(netG(fixed_noise).detach().cpu(), image_size)
                    data = vectors_to_images(real_cpu, image_size)
                    img_grid_fake = vutils.make_grid(fake, normalize=True)
                    img_grid_real = vutils.make_grid(data, normalize=True)

                    writer_fake.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step=epoch
                    )
                    writer_real.add_image(
                        "Mnist Real Images", img_grid_real, global_step=epoch
                    )

            iters += 1

    # Plot loss
    #plot_loss(G_losses, D_losses)

    # Plot some real and generated images
    """
    real_batch = next(iter(dataloader))
    plot_real_images(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)),
                     np.transpose(img_list[-1],(1,2,0)))
    """


def plot_loss(G_losses, D_losses):
    """
    Plot losses over time using matplotlib
    The same can be just observed in tensorboard
    """
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_real_images(real_images, fake_images):
    """
    Plot losses
    The same can be just observed in tensorboard
    """
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(real_images)

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(fake_images)
    plt.show()


if __name__ == '__main__':
    main()
