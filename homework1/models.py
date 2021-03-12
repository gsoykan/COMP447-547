import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# sources
# https://towardsdatascience.com/plotting-probabilities-for-discrete-and-continuous-random-variables-353c5bb62336
def frequencies(values, d):
    frequencies = {}
    for d_i in range(d):
        frequencies[d_i] = 0

    for v in values:
        if v in frequencies:
            frequencies[v] += 1
        else:
            frequencies[v] = 1
    return frequencies


def probabilities(sample, freqs):
    probs = []
    for k, v in freqs.items():
        probs.append(round(v / len(sample), 1))
    return probs


def compute_crossentropyloss_manual(x, y0):
    torch.exp()
    """
    x is the vector of probabilities with shape (batch_size,C)
    y0 shape is the same (batch_size), whose entries are integers from 0 to C-1
    """
    loss = 0.
    n_batch, n_class = x.shape
    # print(n_class)
    for x1, y1 in zip(x, y0):
        class_index = int(y1.item())
        loss = loss + torch.log(torch.exp(x1[class_index]) / (torch.exp(x1).sum()))
    loss = - loss / n_batch
    return loss


# usage
# sample = [0,1,1,1,1,1,2,2,2,2]
# freqs = frequencies(sample)
# probs = probabilities(sample, freqs)
# x_axis = list(set(sample))plt.bar(x_axis, probs)

# sources:
# https://discuss.pytorch.org/t/how-could-i-create-a-module-with-learnable-parameters/28115
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html

class Histogram(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.W = nn.Parameter(torch.zeros(1, d))
        self.W.requires_grad = True

    def loss(self, x):
        batch_size = x.shape[0]
        mock_x = torch.repeat_interleave(self.W, repeats=batch_size, dim=0)
        loss = nn.CrossEntropyLoss()
        input = mock_x
        target = x
        output = loss(input, target)
        return output

    def get_distribution(self):
        softmax = nn.Softmax()
        distribution = softmax(self.W)
        distribution = distribution.cpu().detach().numpy()
        distribution = np.squeeze(distribution)
        return distribution


# https://github.com/Rayhane-mamah/Tacotron-2/issues/155
# https://arxiv.org/pdf/1701.05517.pdf

class MixtureOfLogistics(nn.Module):
    def __init__(self, d, n_mix=4):
        super().__init__()
        self.d = d
        self.n_mix = 4
        # pi, m, s
        # TODO: find better way to initialize
        self.W = torch.randn(n_mix, 3)
        #self.W[:, 2] = torch.tensor(1 / n_mix)
        self.W[:, 1] = self.W[:, 1] * torch.tensor(int(d / 2))
        #self.W[:, 0] = torch.tensor(1 / n_mix)
        self.W = nn.Parameter(self.W)
        self.W.requires_grad = True

    # torch.sigmoid(), torch.clamp(), F.log_softmax()
    def forward(self, x):
        n_mix = self.n_mix

        # create x for each mixture
        x = torch.repeat_interleave(torch.reshape(x, (-1, 1)), repeats=n_mix, dim=1)

        pi = self.W[:, 0]
        softmaxed_pi = torch.softmax(pi, dim=0)
        mean = self.W[:, 1]
        scale = self.W[:, 2]

        offset = 0.5
        centered_mean = x - mean

        cdfminus_arg = (centered_mean - offset) * torch.exp(scale)
        cdfplus_arg = (centered_mean + offset) * torch.exp(scale)

        cdfminus_safe = torch.sigmoid(cdfminus_arg)
        cdfplus_safe = torch.sigmoid(cdfplus_arg)

        raw_px = torch.where(x <= 0, cdfplus_safe,
                             torch.where(x >= self.d - 1, 1 - cdfminus_safe,
                                         cdfplus_safe - cdfminus_safe ))
        px = torch.sum(softmaxed_pi * raw_px, dim=-1)
        return px


    # For the edge case of when $x = 0$,
    # we replace $x-0.5$ by $-\infty$,
    # and for $x = d-1$,
    # we replace $x+0.5$ by $\infty$.

    # https://github.com/Rayhane-mamah/Tacotron-2/blob/d13dbba16f0a434843916b5a8647a42fe34544f5/wavenet_vocoder/models/mixture.py#L44-L49
    # https://bjlkeng.github.io/posts/pixelcnn/
    def loss(self, x):
        torch.autograd.set_detect_anomaly(True)
        n_mix = self.n_mix

        # create x for each mixture
        x = torch.repeat_interleave(torch.reshape(x, (-1, 1)), repeats=n_mix, dim=1)

        pi = self.W[:, 0]
        mean = self.W[:, 1]
        scale = self.W[:, 2]

        offset = 0.5
        centered_mean = x - mean

        cdfminus_arg = (centered_mean - offset) * torch.exp(scale)
        cdfplus_arg = (centered_mean + offset) * torch.exp(scale)

        cdfminus_safe = torch.sigmoid(cdfminus_arg)
        cdfplus_safe = torch.sigmoid(cdfplus_arg)

        # case 1
        softplus = nn.Softplus()
        log_cdfplus = cdfplus_arg - softplus(cdfplus_arg)

        # case 2
        softplus2 = nn.Softplus()
        log_1minus_cdf = -1 * softplus2(cdfminus_arg)

        log_ll = torch.where(x <= 0, log_cdfplus,
                             torch.where(x >= self.d - 1, log_1minus_cdf,
                                         torch.log(torch.maximum(cdfplus_safe - cdfminus_safe, torch.tensor(1e-10)))))
        pre_result = F.log_softmax(pi) + log_ll
        sum_ll = torch.logsumexp(pre_result, dim=-1)
        avg_loss = -1 * torch.sum(sum_ll)
        return avg_loss

    def get_distribution(self):
        x = torch.Tensor(list(range(0, self.d)))
        distribution = self.forward(x)
        distribution = distribution.cpu().detach().numpy()
        return distribution
