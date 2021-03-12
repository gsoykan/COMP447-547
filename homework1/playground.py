from deepul.hw1_helper import *
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trainer
import models


def proto_q(train_data,
            test_data,
            model,
            lr
            ):
    batch_size = 64
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    train_dataloader = data.DataLoader(train_data,
                                       batch_size=batch_size,
                                       shuffle=True)
    test_dataloader = data.DataLoader(test_data,
                                      batch_size=batch_size,
                                      shuffle=True)
    model, losses = trainer.train_model(model, {
        trainer.DataLoaderType.TRAIN: train_dataloader,
        trainer.DataLoaderType.TEST: test_dataloader
    },
                                        epochs=50,
                                        optimizer=optimizer,
                                        batch_size=batch_size
                                        )

    train_loss = map(lambda x: x[trainer.DataLoaderType.TRAIN], losses)
    test_loss = map(lambda x: x[trainer.DataLoaderType.TEST], losses)
    return list(train_loss), list(test_loss), model.get_distribution()


def q1_a(train_data, test_data, d, dset_id):
    batch_size = 64
    model = models.Histogram(d)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_dataloader = data.DataLoader(train_data,
                                       batch_size=batch_size,
                                       shuffle=True)
    test_dataloader = data.DataLoader(test_data,
                                      batch_size=batch_size,
                                      shuffle=True)
    model, losses = trainer.train_model(model, {
        trainer.DataLoaderType.TRAIN: train_dataloader,
        trainer.DataLoaderType.TEST: test_dataloader
    },
                                        epochs=50,
                                        optimizer=optimizer,
                                        batch_size=batch_size
                                        )

    train_loss = map(lambda x: x[trainer.DataLoaderType.TRAIN], losses)
    test_loss = map(lambda x: x[trainer.DataLoaderType.TEST], losses)
    return list(train_loss), list(test_loss), model.get_distribution()


def q1_b(train_data, test_data, d, dset_id):
    model = models.MixtureOfLogistics(d)
    return proto_q(train_data, test_data, model, lr=0.001)

if __name__ == '__main__':
    # train_data, test_data = q1_sample_data_1()
    # d = 20
    # dset_type = 1
    # train_losses, test_losses, distribution = q1_a(train_data, test_data, d, dset_type)

    # q1_save_results(1, 'a', q1_a)
    # q1_save_results(2, 'a', q1_a)

    q1_save_results(1, 'b', q1_b)
    # q1_save_results(2, 'b', q1_b)
