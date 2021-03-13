import random

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import config
from scipy.stats import bernoulli


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
        # self.W[:, 2] = torch.tensor(1 / n_mix)
        self.W[:, 1] = self.W[:, 1] * torch.tensor(int(d / 2))
        # self.W[:, 0] = torch.tensor(1 / n_mix)
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
                                         cdfplus_safe - cdfminus_safe))
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


# TODO: MADE
# SOURCES
# https://bjlkeng.github.io/posts/autoregressive-autoencoders/
# https://www.ritchievink.com/blog/2019/10/25/distribution-estimation-with-masked-autoencoders/

def to_one_hot(labels, d, to_cuda: bool = False):
    flattened_labels = labels.reshape(-1)  # labels.view(-1)
    if config.use_cuda:
        one_hot = torch.FloatTensor(flattened_labels.shape[0], d).cuda()
    else:
        one_hot = torch.FloatTensor(flattened_labels.shape[0], d)
    one_hot.zero_()
    one_hot.scatter_(1, flattened_labels.unsqueeze(1).long(), 1)
    result = one_hot.view(labels.shape[0], -1)
    return result


class MaskedLinear(nn.Linear):
    def __init__(self,
                 input_dimension_coeff,
                 in_features: int,
                 out_features: int,
                 activation,
                 bias: bool = True,
                 is_output_layer: bool = False,
                 output_m=None,
                 input_m=None):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.is_output_layer = is_output_layer
        # self.weight = Parameter(torch.Tensor(out_features, in_features))

        if is_output_layer:
            hidden_m = output_m
        else:
            mask_range_elements = list(range(1, input_dimension_coeff))
            hidden_m = np.zeros(out_features)
            for i in range(input_dimension_coeff):
                m = random.choice(mask_range_elements)
                block = int(out_features / input_dimension_coeff)
                hidden_m[i * block:i * block + block] = m
        self.hidden_m = hidden_m
        self.activation = activation
        # mask creation
        # TODO: POTENTIAL BOTTLENECK this can be done without for loop
        mask = torch.zeros_like(self.weight).cuda()
        for hi, h_m in enumerate(self.hidden_m):
            for ii, i_m in enumerate(input_m):
                if is_output_layer:
                    mask[hi, ii] = 1 if h_m > i_m else 0
                else:
                    mask[hi, ii] = 1 if h_m >= i_m else 0
        self.mask = mask

    def masked_forward(self,
                       input: Tensor,
                       input_m
                       ) -> Tensor:
        masked_weights = torch.mul(self.weight, self.mask)
        preactivated_result = F.linear(input, masked_weights, self.bias)
        if self.is_output_layer:
            result = preactivated_result
        else:
            result = self.activation(preactivated_result)
        return result


class MADE(nn.Module):
    def __init__(self,
                 d,
                 input_dimension_coeff,
                 h,
                 activation_hidden,
                 H=None,
                 W=None):
        super().__init__()
        self.H = H
        self.W = W
        self.input_shape = (input_dimension_coeff,)
        self.d = d
        self.input_dimension_coeff = input_dimension_coeff
        self.input_dim = d * input_dimension_coeff
        mask_range_elements = list(range(1, self.input_dimension_coeff + 1))
        np.random.shuffle(mask_range_elements)
        input_m = np.zeros(self.input_dim)
        for i in range(input_dimension_coeff):
            m = mask_range_elements[i]
            input_m[i * d: i * d + d] = m
        self.input_m = input_m
        # fix input sizes cuz  we use one hot encoding
        self.hidden_layer_1 = MaskedLinear(
            self.input_dimension_coeff,
            self.input_dim,
            h,
            activation=activation_hidden,
            input_m=input_m
        )

        input_m_2 = self.hidden_layer_1.hidden_m

        self.hidden_layer_2 = MaskedLinear(self.input_dimension_coeff,
                                           h,
                                           h,
                                           activation=activation_hidden,
                                           input_m=input_m_2)

        input_m_3 = self.hidden_layer_2.hidden_m

        self.output_layer = MaskedLinear(self.input_dimension_coeff,
                                         h,
                                         self.input_dim,
                                         activation=None,
                                         is_output_layer=True,
                                         output_m=self.input_m,
                                         input_m=input_m_3)

    def forward(self, x):
        # N x D (d * coeff)
        batch_size = x.shape[0]
        input = to_one_hot(x, d=self.d)
        result = self.one_hotted_forward(input)
        logits = result.view(batch_size, self.input_dimension_coeff, self.d)
        permuted_logits = logits.permute(0, 2, 1).contiguous().view(batch_size, self.d, *self.input_shape)
        return permuted_logits

    def one_hotted_forward(self, input):
        l1_res = self.hidden_layer_1.masked_forward(input,
                                                    input_m=self.input_m)
        l2_res = self.hidden_layer_2.masked_forward(l1_res,
                                                    input_m=self.hidden_layer_1.hidden_m)
        output = self.output_layer.masked_forward(l2_res,
                                                  input_m=self.hidden_layer_2.hidden_m)
        return output

    # sampling functionında softmax kullanabiliriz
    # first forward then softmax

    def loss(self, x):
        forward_result = self.forward(x)
        loss_fn = nn.CrossEntropyLoss()
        reshaped_x = torch.reshape(x, (x.shape[0], -1)).long()
        total_loss = loss_fn(forward_result, reshaped_x)
        return total_loss

    def get_distribution(self):
        assert self.input_shape == (2,), 'Only available for 2D joint'
        x = np.mgrid[0:self.d, 0:self.d].reshape(2, self.d ** 2).T
        x = torch.LongTensor(x).cuda()
        log_probs = F.log_softmax(self(x), dim=1)
        distribution = torch.gather(log_probs, 1, x.unsqueeze(1)).squeeze(1)
        distribution = distribution.sum(dim=1)
        return distribution.exp().view(self.d, self.d).detach().cpu().numpy()

    def generate_sample(self):
        initial_batch = torch.zeros((100, self.H, self.W, 1)).to(config.device)
        for i in range(self.input_dimension_coeff):
            active_index = int((np.where(self.input_m == i + 1)[0] / 2)[0])
            forward_result = self.forward(initial_batch)
            raw_p = forward_result[:, :, active_index]
            probs = torch.softmax(raw_p, dim=1)
            probs_for_1 = probs[:, 1]
            new_values = []
            for p_1 in probs_for_1:
                p_1_numpy = p_1.detach().cpu().numpy()
                new_value = bernoulli.rvs(p_1_numpy)
                new_values.append(new_value)
            new_values_tensor = Tensor(new_values)
            new_values_tensor = torch.reshape(new_values_tensor, (-1, 1))
            second_index = active_index % self.W
            first_index = active_index // self.W
            initial_batch[:, first_index, second_index] = new_values_tensor

        return initial_batch.detach().cpu().numpy()
