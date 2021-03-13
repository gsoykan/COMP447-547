import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum
import models
import config

# Hint: You may want to implement training and the evaluation procedures as functions
# which take a model and the dataloaders as an input and return the losses.


class DataLoaderType(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


data.DataLoader


def train_model(model,
                # {DataLoaderType: DL}
                dataloader_dict,
                epochs,
                optimizer,
                batch_size):
    epoch_losses = []
    train_iteration_losses = []
    dataloader_types = dataloader_dict.keys()
    for epoch in range(-1, epochs):  # loop over the dataset multiple times
        losses = {}
        for dataloader_type in dataloader_types:
            if epoch == -1 and dataloader_type == DataLoaderType.TRAIN:
                continue
            losses[dataloader_type] = 0
            dataloader = dataloader_dict[dataloader_type]
            num_batches = len(dataloader)
            for i, data in enumerate(dataloader, 0):
                inputs = data.to(config.device)
                if dataloader_type == DataLoaderType.TRAIN:
                    model.train()
                    optimizer.zero_grad()
                    loss = model.loss(inputs)
                    loss.backward()
                    optimizer.step()
                    train_iteration_losses.append(loss.item())
                elif dataloader_type == DataLoaderType.VALIDATION:
                    model.eval()
                    loss = model.loss(inputs)
                elif dataloader_type == DataLoaderType.TEST:
                    model.eval()
                    loss = model.loss(inputs)
                losses[dataloader_type] += loss.item()

                if i % 50 == 0:
                    print(str(dataloader_type) + " " + '[%d, %5d] loss: %.5f' %
                          (epoch + 1, i + 1, loss))

            losses[dataloader_type] /= num_batches
        epoch_losses.append(losses)
    return model, epoch_losses, train_iteration_losses
