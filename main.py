import numpy as np
from common.dataset import *
from common.model import *
from common.metric import Metric


def train_model(epochs=5):
    """

    :param epochs:
    :return:
    """

    train_data_dir = '/data/content/training_set/training_set'
    test_data_dir = '/data/content/test_set/test_set'

    trn_dl, val_dl = get_data(train_data_dir, test_data_dir)

    model = Model()
    model, loss_fn, optimizer = model.get_model()

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        train_epoch_losses, train_epoch_accuracies = [], []
        val_epoch_accuracies = []
        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            batch_loss = train_batch(x, y, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss)

        train_epoch_loss = np.array(train_epoch_losses).mean()

        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            is_correct = Metric.accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)

        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        for ix, batch in enumerate(iter(val_dl)):
            x, y = batch
            val_is_correct = Metric.accuracy(x, y, model)
            val_epoch_accuracies.extend(val_is_correct)

        val_epoch_accuracy = np.mean(val_epoch_accuracies)

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_accuracies.append(val_epoch_accuracy)

    return train_losses, train_accuracies, val_accuracies


if __name__ == '__main__':
    train_losses, train_accuracies, val_accuracies = train_model(epochs=100)

