import os
from incense import ExperimentLoader
import matplotlib.pyplot as plt
from matplotlib import rc
import math

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})


def get_uri_db_pair():
    uri = os.environ.get('MLAB_URI')
    database = os.environ.get('MLAB_DB')
    if all([uri, database]):
        return uri, database
    else:
        raise ConnectionError('Could not find URI or database')


def get_experiment(exp_id):
    uri, database = get_uri_db_pair()
    loader = ExperimentLoader(mongo_uri=uri, db_name=database)
    ex = loader.find_by_id(exp_id)
    return ex


def plot_elbo(exp_id):
    ex = get_experiment(exp_id)
    train_elbo = ex.metrics['train_elbo']
    val_elbo = ex.metrics['val_elbo']
    plt.figure(figsize=(3.7, 2.7))
    plt.plot(train_elbo, label='train')
    plt.plot(val_elbo, label='val')
    plt.legend()
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.grid()
    plt.show()


def plot_rec_kl_losses(exp_id):
    ex = get_experiment(exp_id)
    train_rec = ex.metrics['train_rec']
    train_kl = ex.metrics['train_kl']
    plt.figure(figsize=(3.7, 2.7))
    plt.plot(train_rec, label=r'$\mathcal{L}^{recon}$')
    plt.plot(train_kl, label=r'$\mathcal{L}^{reg}$')
    plt.legend()
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.grid()
    plt.show()


def plot_gan_curves(exp_id, plot_optima=False):
    ex = get_experiment(exp_id)
    loss_d = ex.metrics['loss_d']
    loss_g = ex.metrics['loss_g']
    plt.figure(figsize=(3.7, 2.7))
    plt.plot(loss_d, label=r'$\mathcal{L}^{D}$')
    plt.plot(loss_g, label=r'$\mathcal{L}^{G}$')

    # Optimal values
    if plot_optima:
        interval = [min(loss_d.index), max(loss_d.index)]
        plt.plot(interval, 2 * [math.log(4)], 'k--', alpha=0.5)
        plt.plot(interval, 2 * [math.log(2)], 'k--', alpha=0.5)

    plt.legend(loc='upper right')
    plt.xlabel('Iterations')
    plt.tight_layout()
    plt.grid()
    plt.show()


def plot_nf_curves(exp_id):
    ex = get_experiment(exp_id)
    train_bpd = ex.metrics['train_bpd']
    val_bpd = ex.metrics['val_pbd']
    plt.figure(figsize=(5.4, 3.0))
    plt.plot(train_bpd, label='train')
    plt.plot(val_bpd, label='val')

    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('bpd')
    plt.tight_layout()
    plt.grid()
    plt.show()


# plot_elbo(7)
# plot_rec_kl_losses(7)
# plot_gan_curves(12)
# plot_gan_curves(14, plot_optima=True)
plot_nf_curves(32)
