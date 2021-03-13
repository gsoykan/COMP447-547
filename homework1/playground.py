from deepul.hw1_helper import *
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trainer
import models
import config



def proto_q(train_data,
            test_data,
            model,
            lr,
            epochs=50,
            final_func = None
            ):
    batch_size = 64
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    train_dataloader = data.DataLoader(train_data,
                                       batch_size=batch_size,
                                       shuffle=True)
    test_dataloader = data.DataLoader(test_data,
                                      batch_size=batch_size,
                                      shuffle=True)
    model, losses, train_iteration_losses = trainer.train_model(model, {
        trainer.DataLoaderType.TRAIN: train_dataloader,
        trainer.DataLoaderType.TEST: test_dataloader
    },
                                                                epochs=epochs,
                                                                optimizer=optimizer,
                                                                batch_size=batch_size
                                                                )

    # train_loss = map(lambda x: x[trainer.DataLoaderType.TRAIN], losses)
    train_loss = train_iteration_losses
    test_loss = map(lambda x: x[trainer.DataLoaderType.TEST], losses)
    final_res = final_func() if final_func is not None else model.get_distribution()
    #distribution = np.full((25, 25), 1/(25*25))
    return list(train_loss), list(test_loss), final_res


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
    model, losses, train_iteration_losses = trainer.train_model(model, {
        trainer.DataLoaderType.TRAIN: train_dataloader,
        trainer.DataLoaderType.TEST: test_dataloader
    },
                                                                epochs=50,
                                                                optimizer=optimizer,
                                                                batch_size=batch_size
                                                                )

    # train_loss = map(lambda x: x[trainer.DataLoaderType.TRAIN], losses)
    train_loss = train_iteration_losses
    test_loss = map(lambda x: x[trainer.DataLoaderType.TEST], losses)
    return list(train_loss), list(test_loss), model.get_distribution()


def q1_b(train_data, test_data, d, dset_id):
    model = models.MixtureOfLogistics(d)
    return proto_q(train_data, test_data, model, lr=0.001)


def q2_a(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train, 2) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test, 2) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for each random variable x1 and x2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d, d) of probabilities (the learned joint distribution)
    """

    model = models.MADE(d=d,
                        h=64,
                        activation_hidden=nn.ELU(),
                        input_dimension_coeff=2)
    model.to(config.device)
    return proto_q(train_data,
                   test_data,
                   model,
                   lr=0.1,
                   epochs=20)


def q2_b(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: An (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    image_shape: (H, W), height and width of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """

    _, H, W, _ = train_data.shape

    model = models.MADE(d=2,
                        h=64,
                        activation_hidden=nn.ELU(),
                        input_dimension_coeff=H*W)
    model.to(config.device)
    return proto_q(train_data,
                   test_data,
                   model,
                   lr=0.1,
                   epochs=5)

if __name__ == '__main__':
    print("program start")
    # train_data, test_data = q1_sample_data_1()
    # d = 20
    # dset_type = 1
    # train_losses, test_losses, distribution = q1_a(train_data, test_data, d, dset_type)

    # q1_save_results(1, 'a', q1_a)
    # q1_save_results(2, 'a', q1_a)

    # q1_save_results(1, 'b', q1_b)
    # q1_save_results(2, 'b', q1_b)

    # Final Test Loss: 3.1860
    #q2_save_results(1, 'a', q2_a)

    # Final Test Loss: 5.2991
    #q2_save_results(2, 'a', q2_a)

    #q2_save_results(1, 'b', q2_b)
    #q2_save_results(2, 'b', q2_b)