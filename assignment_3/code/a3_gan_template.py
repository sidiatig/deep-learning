import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
# Set up database logs
uri = os.environ.get('MLAB_URI')
database = os.environ.get('MLAB_DB')
if all([uri, database]):
    print(uri)
    print(database)
    ex.observers.append(MongoObserver.create(uri, database))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_PIXELS = IMG_WIDTH * IMG_HEIGHT


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(nn.Linear(args.latent_dim, 128),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(128, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(256, 512),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(512, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(1024, IMG_PIXELS),
                                    nn.Tanh())

    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(nn.Linear(IMG_PIXELS, 512),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Dropout(0.3),
                                    nn.Linear(512, 256),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Dropout(0.3),
                                    nn.Linear(256, 1))

    def forward(self, img):
        return self.layers(img)


def sample_generator(generator, n_samples, z=None):
    """Obtain samples from the generator. The returned tensor is on device and
    attached to the graph, so it has requires_grad=True """
    if z is None:
        z = torch.randn(n_samples, args.latent_dim).to(device)

    samples = generator(z)
    return samples


@ex.capture
def save_samples(generator, fname, _run):
    samples = sample_generator(generator, n_samples=25).detach().cpu()
    samples = samples.reshape(-1, 1, IMG_HEIGHT, IMG_WIDTH) * 0.5 + 0.5

    grid = make_grid(samples, nrow=5)[0]
    plt.cla()
    plt.imshow(grid.numpy(), cmap='binary')
    plt.axis('off')
    img_path = os.path.join(os.path.dirname(__file__), 'saved', fname)
    plt.savefig(img_path)
    _run.add_artifact(img_path, fname)
    os.remove(img_path)


@ex.capture
def train(dataloader, discriminator, generator, optimizer_g, optimizer_d, _run):
    ones = torch.ones((args.batch_size, 1), dtype=torch.float32).to(device)
    zeros = torch.zeros((args.batch_size, 1), dtype=torch.float32).to(device)
    bce_loss = nn.BCEWithLogitsLoss()

    train_iters = 0
    avg_loss_d = 0
    avg_loss_g = 0
    log = 'epoch [{:d}/{:d}] batch [{:d}/{:d}] loss_d: {:.6f} loss_g: {:.6f}'
    n_epochs = args.n_epochs

    for epoch in range(1, n_epochs + 1):
        if epoch == 1 or epoch % args.save_epochs == 0:
            fname = 'samples_{:d}.png'.format(epoch)
            save_samples(generator, fname)

        for i, (imgs, _) in enumerate(dataloader):
            # Train Discriminator
            # -------------------
            imgs = imgs.reshape(args.batch_size, -1).to(device)
            samples = sample_generator(generator, args.batch_size).detach()
            optimizer_d.zero_grad()
            pos_preds = discriminator(imgs)
            neg_preds = discriminator(samples)
            # One-sided label smoothing
            ones.uniform_(0.7, 1.2)
            loss_d = bce_loss(pos_preds, ones) + bce_loss(neg_preds, zeros)
            loss_d.backward()
            optimizer_d.step()
            # -------------------

            # Train Generator
            # -------------------
            samples = sample_generator(generator, args.batch_size)
            optimizer_g.zero_grad()
            neg_preds = discriminator(samples)
            loss_g = bce_loss(neg_preds, ones)
            loss_g.backward()
            optimizer_g.step()
            # -------------------

            train_iters += 1
            avg_loss_d += loss_d.item() / args.log_interval
            avg_loss_g += loss_g.item() / args.log_interval

            if train_iters % args.log_interval == 0:
                print(log.format(epoch + 1, n_epochs,
                                 i + 1, len(dataloader),
                                 avg_loss_d, avg_loss_g))
                _run.log_scalar('loss_d', avg_loss_d, train_iters)
                _run.log_scalar('loss_g', avg_loss_g, train_iters)
                avg_loss_d = 0
                avg_loss_g = 0


@ex.main
def main(timestamp):
    # Load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))])),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=1)

    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Train GAN
    train(dataloader, discriminator, generator, optimizer_g, optimizer_d)

    # Save generator
    fname = str(timestamp) + '.pt'
    model_path = os.path.join(os.path.dirname(__file__), 'saved', fname)
    torch.save(generator.state_dict(), model_path)
    print('Saved generator to {}'.format(model_path))


def load_generator(model_path):
    generator = Generator()
    generator.load_state_dict(torch.load(model_path, map_location='cpu'))
    generator.eval()
    return generator


@ex.command
def interpolate(model_path, steps=20, method='slerp'):
    """Generate samples from an interpolation of two latent values"""
    generator = load_generator(model_path)

    torch.manual_seed(29)
    z1 = torch.randn(1, args.latent_dim)
    torch.manual_seed(6)
    z2 = torch.randn(1, args.latent_dim)

    n_images = steps + 2
    coeffs = torch.linspace(0, 1, n_images).unsqueeze(dim=1)

    if method == 'lerp':
        z_interp = (1 - coeffs) * z1 + coeffs * z2
    elif method == 'slerp':
        angle = torch.acos(torch.cosine_similarity(z1, z2, dim=1))
        sin_angle = torch.sin(angle)
        z_interp = (torch.sin((1.0 - coeffs) * angle) / sin_angle * z1
                    + torch.sin(coeffs * angle) / sin_angle * z2)
    else:
        raise ValueError(f'Unknown method {method}')

    x = generator(z_interp).detach().cpu()
    x = x.reshape(-1, 1, IMG_HEIGHT, IMG_WIDTH) * 0.5 + 0.5
    grid = make_grid(x, nrow=n_images)[0]

    plt.figure(figsize=(5.0, 1.0))
    plt.imshow(grid.numpy(), cmap='binary')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


@ex.command
def explore(model_path, n_samples=20):
    """Sample the generator with different random seeds.
    Each sample is shown one after the other until n_samples are shown.
    Use it to find the right sandom seed to sample a specific digit."""
    generator = load_generator(model_path)

    for i in range(n_samples):
        torch.manual_seed(i)
        sample = sample_generator(generator, n_samples=1).detach().cpu()
        sample = sample.reshape(IMG_WIDTH, IMG_HEIGHT) * 0.5 + 0.5

        plt.figure(figsize=(2.0, 2.0))
        plt.imshow(sample.numpy(), cmap='binary')
        plt.axis('off')
        plt.title(f'Seed: {i:d}')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, choices=['main',
                                                        'interpolate',
                                                        'explore'],
                        default='main')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_epochs', type=int, default=20,
                        help='save samples every SAVE_EPOCHS epochs')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='log every LOG_INTERVAL iterations')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path of a saved generator')
    args = parser.parse_args()

    # noinspection PyUnusedLocal
    @ex.config
    def config():
        timestamp = int(datetime.now().timestamp())
        model_path = args.model_path

    ex.run(args.command)
