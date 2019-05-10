import os
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist

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

IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_PIXELS = IMG_WIDTH * IMG_HEIGHT

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.linear = nn.Linear(in_features=IMG_PIXELS,
                                out_features=hidden_dim)
        self.linear_mu = nn.Linear(in_features=hidden_dim,
                                   out_features=z_dim)
        self.linear_logvar = nn.Linear(in_features=hidden_dim,
                                       out_features=z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        h = torch.relu(self.linear(input))
        mean = self.linear_mu(h)
        logvar = self.linear_logvar(h)

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.layers = nn.Sequential(nn.Linear(in_features=z_dim,
                                              out_features=hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(in_features=hidden_dim,
                                              out_features=IMG_PIXELS))

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.layers(input)
        return mean


class VAE(nn.Module):
    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)
        self.rec_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, logvar = self.encoder(input)
        z = mean + torch.randn_like(mean) * torch.sqrt(torch.exp(logvar))
        x_rec = self.decoder(z)

        rec_loss = self.rec_loss(x_rec, input).sum(dim=-1)
        kl = 0.5 * (torch.exp(logvar) + mean**2 - logvar - 1).sum(dim=-1)

        return rec_loss.mean(), kl.mean()

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        z = torch.randn((n_samples, self.z_dim)).to(device)
        im_means = torch.sigmoid(self.decoder(z))
        sampled_ims = torch.bernoulli(im_means)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0

    for imgs in data:
        imgs = imgs.to(device)
        rec_loss, kl = model(imgs.view(-1, IMG_PIXELS))
        avg_elbo = rec_loss + kl
        average_epoch_elbo += avg_elbo.item()

        if model.training:
            optimizer.zero_grad()
            avg_elbo.backward()
            optimizer.step()

    average_epoch_elbo /= len(data)

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    with torch.no_grad():
        model.eval()
        val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


@ex.capture
def save_samples(model, fname, _run):
    samples, _ = model.sample(n_samples=16)
    samples = samples.detach().cpu()
    grid = make_grid(samples.reshape(-1, 1, IMG_WIDTH, IMG_HEIGHT),
                     nrow=4)
    plt.cla()
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    img_path = os.path.join(os.path.dirname(__file__), 'saved', fname)
    plt.savefig(img_path)
    _run.add_artifact(img_path, fname)
    os.remove(img_path)


@ex.main
def main(epochs, zdim, _run):
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=zdim).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, epochs + 1):
        # Save samples at beginning, 50% and 100% of training
        if int(100 * epoch/epochs) in [int(100/epochs), 50, 100]:
            fname = 'samples_{:d}.png'.format(epoch)
            save_samples(model, fname)

        elbos = run_epoch(model, data, optimizer)

        train_elbo, val_elbo = elbos
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")
        _run.log_scalar('train_elbo', train_elbo, epoch)
        _run.log_scalar('val_elbo', val_elbo, epoch)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    args = parser.parse_args()

    @ex.config
    def config():
        epochs = args.epochs
        zdim = args.epochs

    ex.run()
