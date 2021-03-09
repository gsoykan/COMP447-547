from deepul.hw1_helper import *
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trainer
import models


def q1_a(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test,) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for random variable x
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d,) of model probabilities
    """
    batch_size = 64
    model = models.Histogram(d)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_dataloader = data.DataLoader(train_data,
                                       batch_size=batch_size,
                                       shuffle=True)
    test_dataloader = data.DataLoader(test_data,
                                      batch_size=batch_size,
                                      shuffle=True)
    trainer.train_model(model, {
        trainer.DataLoaderType.TRAIN: train_dataloader,
        trainer.DataLoaderType.TEST: test_dataloader
    },
                        epochs=10,
                        optimizer=optimizer
                        )
    return [100], [100], model.get_distribution()


if __name__ == '__main__':
    # train_data, test_data = q1_sample_data_1()
    # d = 20
    # dset_type = 1
    # train_losses, test_losses, distribution = q1_a(train_data, test_data, d, dset_type)
    q1_save_results(1, 'a', q1_a)
