from torch import nn
import torch


torch.manual_seed(1)



class Discriminator_FCNN(nn.Module):


    def __init__(self, num_input_features, ngpu):
        """
        This class describes the FCNN discriminator
        as described in the paper https://arxiv.org/pdf/2010.06201.pdf 
        """
        super(Discriminator_FCNN, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # Paper doesn't specify if bias or batchnorm are used
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(num_input_features, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # First hidden layer (64 -> 16)
            nn.Linear(64, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second hidden layer (16 -> output)
            nn.Linear(16, 1, bias=False),
            nn.Sigmoid()
        )

    
    def forward(self, input):
        return self.main(input)


class DiscriminatorNet(nn.Module):


    def __init__(self, n_features, ngpu):
        """
        A three hidden-layer discriminative neural network
        Source: medium article
        """
        super(DiscriminatorNet, self).__init__()
        self.ngpu = ngpu
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    