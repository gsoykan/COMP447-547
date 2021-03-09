import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum
import models

# Hint: You may want to implement training and the evaluation procedures as functions
# which take a model and the dataloaders as an input and return the losses.


class DataLoaderType(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


def train_model(model,
                # {DataLoaderType: DL}
                dataloader_dict,
                epochs,
                optimizer):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        dataloader_types = dataloader_dict.keys()
        for dataloader_type in dataloader_types:
            dataloader = dataloader_dict[dataloader_type]
            for i, data in enumerate(dataloader, 0):
                inputs = data
                if dataloader_type == DataLoaderType.TRAIN:
                    model.train()
                    optimizer.zero_grad()
                    loss = model.loss(inputs)
                    loss.backward()
                    optimizer.step()
                elif dataloader_type == DataLoaderType.VALIDATION:
                    model.eval()
                    loss = model.loss(inputs)
                elif dataloader_type == DataLoaderType.TEST:
                    model.eval()
                    loss = model.loss(inputs)
                running_loss += loss.item()
                if i % 50 == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
