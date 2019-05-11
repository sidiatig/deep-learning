import os
from incense import ExperimentLoader
import matplotlib.pyplot as plt


def get_uri_db_pair():
    uri = os.environ.get('MLAB_URI')
    database = os.environ.get('MLAB_DB')
    if all([uri, database]):
        return uri, database
    else:
        raise ConnectionError('Could not find URI or database')


def get_experiment(id):
    uri, database = get_uri_db_pair()
    loader = ExperimentLoader(mongo_uri=uri, db_name=database)
    ex = loader.find_by_id(id)
    return ex


def plot_elbo(id):
    ex = get_experiment(id)
    train_elbo = ex.metrics['train_elbo']
    val_elbo = ex.metrics['val_elbo']
    plt.figure(figsize=(4.0, 3.0))
    plt.plot(train_elbo, label='train')
    plt.plot(val_elbo, label='val')
    plt.legend()
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.show()


def plot_rec_kl_losses(id):
    ex = get_experiment(id)
    train_rec = ex.metrics['train_rec']
    train_kl = ex.metrics['train_kl']
    plt.figure(figsize=(4.0, 3.0))
    plt.plot(train_rec, label=r'$\mathcal{L}^{recon}$')
    plt.plot(train_kl, label=r'$\mathcal{L}^{reg}$')
    plt.legend()
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.show()
