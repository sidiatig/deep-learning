import argparse
from datetime import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datasets.mnist import mnist
import os
from torchvision.utils import make_grid

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


def log_prior(x):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """
    logp = -0.5 * ((x ** 2) + np.log(2.0 * np.pi))

    return logp


def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """
    sample = torch.randn(size)

    return sample


def get_mask():
    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28*28)
    mask = torch.from_numpy(mask)

    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024, mean_only=False):
        super().__init__()
        self.n_hidden = n_hidden

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.
        # Suggestion: Linear ReLU Linear ReLU Linear.
        if mean_only:
            out_features = c_in
        else:
            out_features = 2

        self.mean_only = mean_only
        self.nn = torch.nn.Sequential(nn.Linear(c_in, n_hidden),
                                      nn.ReLU(),
                                      nn.Linear(n_hidden, n_hidden),
                                      nn.ReLU(),
                                      nn.Linear(n_hidden, out_features))

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.
        mask = self.mask
        neg_mask = 1 - mask
        loc_scale = self.nn(z * mask)

        if self.mean_only:
            loc = loc_scale

            if not reverse:
                z = mask * z + neg_mask * (z + loc)
            else:
                z = mask * z + neg_mask * (z - loc)
        else:
            loc, log_scale = torch.chunk(loc_scale, chunks=2, dim=1)
            log_scale = torch.tanh(log_scale)

            if not reverse:
                z = mask * z + neg_mask * (z * torch.exp(log_scale) + loc)
                ldj += (neg_mask * log_scale).sum(dim=1)
            else:
                z = mask * z + neg_mask * (z - loc) * torch.exp(-log_scale)
                ldj += (neg_mask * -log_scale).sum(dim=1)

        return z, ldj


class Flow(nn.Module):
    def __init__(self, shape, n_flows=4):
        super().__init__()
        channels, = shape

        mask = get_mask()

        self.layers = torch.nn.ModuleList()

        for i in range(n_flows):
            self.layers.append(Coupling(c_in=channels, mask=mask))
            self.layers.append(Coupling(c_in=channels, mask=1-mask))

        self.z_shape = (channels,)

    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z, logdet


class Model(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.flow = Flow(shape)

    def dequantize(self, z):
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            # Divide by 256 and update ldj.
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z*(1-alpha) + alpha*0.5
            logdet += torch.sum(-torch.log(z) - torch.log(1-z), dim=1)
            z = torch.log(z) - torch.log(1-z)

        else:
            # Inverse normalize
            logdet += torch.sum(torch.log(z) + torch.log(1-z), dim=1)
            z = torch.sigmoid(z)

            # Multiply by 256.
            z = z * 256.
            logdet += np.log(256) * np.prod(z.size()[1:])

        return z, logdet

    def forward(self, input):
        """
        Given input, encode the input to z space. Also keep track of ldj.
        """
        z = input
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)

        # Compute log_pz and log_px per example
        log_pz = log_prior(z).sum(dim=1)
        log_px = log_pz + ldj

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """
        z = sample_prior((n_samples,) + self.flow.z_shape).to(device)
        ldj = torch.zeros(z.size(0)).to(device)

        z, ldj = self.flow(z, ldj, reverse=True)
        z, ldj = self.logit_normalize(z, ldj, reverse=True)

        return z


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """

    avg_bpd = 0

    for imgs, labels in data:
        imgs = imgs.to(device)
        log_px = model(imgs)
        loss = -log_px.mean()

        bpd = loss.item() / (np.log(2) * IMG_PIXELS)
        avg_bpd += bpd

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

    avg_bpd /= len(data)
    return avg_bpd


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer)

    model.eval()
    with torch.no_grad():
        val_bpd = epoch_iter(model, valdata, optimizer)

    return train_bpd, val_bpd


@ex.capture
def save_samples(model, fname, _run):
    samples = model.sample(n_samples=16).detach().cpu()
    samples = samples.reshape(-1, 1, IMG_WIDTH, IMG_HEIGHT)
    grid = make_grid(samples, nrow=4, normalize=True)[0]

    plt.cla()
    plt.imshow(grid.numpy(), cmap='binary')
    plt.axis('off')
    img_path = os.path.join(os.path.dirname(__file__), 'saved', fname)
    plt.savefig(img_path)
    _run.add_artifact(img_path, fname)
    os.remove(img_path)


@ex.main
def main(epochs, timestamp, _run):
    data = mnist()[:2]  # ignore test split

    model = Model(shape=[IMG_PIXELS]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        # Save samples at beginning, 50% and 100% of training
        if int(100 * epoch / epochs) in [int(100 / epochs), 50, 100]:
            fname = 'nf_{:d}.png'.format(epoch)
            save_samples(model, fname)

        bpds = run_epoch(model, data, optimizer)
        train_bpd, val_bpd = bpds
        
        print("[Epoch {epoch}] train bpd: {train_bpd} val_bpd: {val_bpd}".format(
            epoch=epoch, train_bpd=train_bpd, val_bpd=val_bpd))
        _run.log_scalar('train_bpd', train_bpd, epoch)
        _run.log_scalar('val_pbd', val_bpd, epoch)

    # Save model
    fname = str(timestamp) + '.pt'
    model_path = os.path.join(os.path.dirname(__file__), 'saved', fname)
    torch.save(model.state_dict(), model_path)
    print('Saved model to {}'.format(model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')

    args = parser.parse_args()

    # noinspection PyUnusedLocal
    @ex.config
    def config():
        epochs = args.epochs
        timestamp = int(datetime.now().timestamp())

    ex.run()
