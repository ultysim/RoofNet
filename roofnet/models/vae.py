import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, img_dim=32, image_channels=3, h_dim=1024, z_dim=128, device=None):
        super(VAE, self).__init__()

        assert img_dim == 32 or img_dim == 64 or img_dim == 255, 'img_dim must be 32 or 64 or 255'
        if img_dim == 32:
            self.encoder = nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 256, kernel_size=4, stride=2),
                nn.ReLU(),
                Flatten()
            ).to(device)

            self.decoder = nn.Sequential(
                UnFlatten(),
                nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),#1x1 -> 5x5
                nn.ReLU(),
                nn.ConvTranspose2d(128, 32, kernel_size=6, stride=2), #5x5 -> 14x14
                nn.ReLU(),
                nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2), #14x14 -> 32x32
                nn.Sigmoid(),
            ).to(device)

        elif img_dim == 64:
            self.encoder = nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size=4, stride=2), #64x64 -> 31x31
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), #31x31 -> 14x14
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2), #14x14 -> 6x6
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2), #6x6 -> 2x2: 4x256=1024
                nn.ReLU(),
                Flatten()
            ).to(device)

            self.decoder = nn.Sequential(
                UnFlatten(),
                nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
                nn.Sigmoid(),
            ).to(device)

        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(image_channels, 8, kernel_size=3, stride=2), #255x255 -> 127x127
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2),  #129x129 -> 63x63
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),  #63x63 -> 31x31
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), #31x31 - > 14x14
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2), #14x14 -> 6x6
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2), #6x6 -> 2x2, needs to be output of 2x2 for zdim
                nn.ReLU(),
                Flatten()
            ).to(device)

            self.decoder = nn.Sequential(
                UnFlatten(),
                nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2), #1x1 -> 5x5
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), #5x5 -> 13x13
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2), #13x13 -> 29x29
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16,  kernel_size=5, stride=2), #29x29 -> 61x61
                nn.ReLU(),
                nn.ConvTranspose2d(16, 8,  kernel_size=6, stride=2), #61x61 -> 126x126
                nn.ReLU(),
                nn.ConvTranspose2d(8, image_channels,  kernel_size=5, stride=2), #126x126  -> 255x255
                nn.Sigmoid(),
            ).to(device)

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.device = device

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)

        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)

        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z, mu, logvar
