import sys
import numpy as np
import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(3, hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(3, hidden_size,
                                                   hidden_size))
        self.bias = nn.Parameter(torch.Tensor(3, hidden_size))

    def forward(self, input, hx=None):
        weight_ih = self.weight_ih
        bias = self.bias
        weight_hh = self.weight_hh

        z = (torch.mv(weight_ih[0], input) + torch.mv(weight_hh[0], hx) +
             bias[0]).sigmoid()
        r = (torch.mv(weight_ih[1], input) + torch.mv(weight_hh[1], hx) +
             bias[1]).sigmoid()
        n = (torch.mv(weight_ih[2], input) + torch.mv(weight_hh[2], hx * r) +
             bias[2]).relu()
        h = (torch.ones_like(z) - z) * n + z * hx

        return h


class PoolingGRU(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 pooling_size,
                 gaussian_num,
                 hidden_dim=128,
                 is_training=True):
        super(PoolingGRU, self).__init__()
        self.gaussian_num = gaussian_num
        self.hidden_dim = hidden_dim
        self.is_training = is_training

        self.xy_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True))

        self.hidden_embed = nn.Sequential(
            nn.Linear(pooling_size * pooling_size * hidden_dim,
                      hidden_dim,
                      bias=False), nn.ReLU(inplace=True))

        self.gru = GRUCell(hidden_dim * 2, hidden_dim)
        self.predict_xy_linear = nn.Linear(hidden_dim, output_dim)
        self.predict_gaussian_linear = nn.Linear(hidden_dim,
                                                 output_dim * gaussian_num)

        self.init_weights()

    def forward(self, xy, pooled_hidden, prev_hidden):
        hidden_dim = self.hidden_dim

        x_e = self.xy_embed(xy)  # (2,) -> (128,)
        H_e = self.hidden_embed(pooled_hidden.view(
            -1, ))  # (20,20,128) -> (20*20*128,) -> (128,)

        cat = torch.cat([x_e, H_e])  # -> (256,)
        hidden = self.gru(cat, prev_hidden)  # (256,) (128,) -> (128,)

        if self.is_training:
            predict = self.predict_gaussian_linear(hidden)
            gaussian_params = self.get_gaussian_params(predict)
            return [gaussian_params, hidden]
        else:
            predict = self.predict_xy_linear(hidden)  # (128,) -> (2,)
            return [predict, hidden]

    def init_weights(self):
        optim_range = np.sqrt(1. / self.hidden_dim)

        self.xy_embed[0].weight.data.uniform_(-optim_range, optim_range)
        self.hidden_embed[0].weight.data.uniform_(-optim_range, optim_range)

        self.gru.weight_ih.data.uniform_(-optim_range, optim_range)
        self.gru.weight_hh.data.uniform_(-optim_range, optim_range)
        self.gru.bias.data.zero_()

        self.predict_xy_linear.weight.data.uniform_(-optim_range, optim_range)
        self.predict_xy_linear.bias.data.zero_()

        self.predict_gaussian_linear.weight.data.uniform_(
            -optim_range, optim_range)
        self.predict_gaussian_linear.bias.data.zero_()

    def get_gaussian_params(self, predict):
        pi, u_x, u_y, sigma_x, sigma_y, rho_xy = torch.split(predict,
                                                             self.gaussian_num,
                                                             dim=0)
        pi = nn.Softmax(dim=0)(pi)
        sigma_x = torch.exp(sigma_x)
        sigma_y = torch.exp(sigma_y)
        rho_xy = torch.tanh(rho_xy)

        return (pi, u_x, u_y, sigma_x, sigma_y, rho_xy)


class Attention(nn.Module):
    def __init__(self, hidden_dim, atten_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.atten_dim = atten_dim

        self.temporal_edge_layer = nn.Linear(hidden_dim, atten_dim)
        self.spatial_edge_layer = nn.Linear(hidden_dim, atten_dim)

    def forward(self, temp_hidden, spat_hidden):
        num_edges = spat_hidden.shape[0]

        # Embed the temporal edgeRNN hidden state
        temporal_embed = self.temporal_edge_layer(
            temp_hidden)  # (hidden_dim,) -> (attn_dim,)

        # Embed the spatial edgeRNN hidden states
        spatial_embed = self.spatial_edge_layer(
            spat_hidden)  # (num_edges, hidden_dim) -> (num_edges, attn_dim)

        # Dot based attention
        if num_edges > 1:
            attn = torch.mv(spatial_embed, temporal_embed)  # (num_edges,)
        else:
            attn = torch.dot(spatial_embed.view(-1), temporal_embed)  # (1,)

        # Variable length
        temperature = num_edges / np.sqrt(self.atten_dim)
        attn = torch.mul(attn, temperature)

        # Softmax
        weighted_value = nn.functional.softmax(attn, dim=0)

        # Compute weighted value
        H = torch.mv(torch.t(spat_hidden), weighted_value.view(
            -1))  # (num_edges, hidden_dim) (num_edges,) ->  (hidden_dim,)

        return H


class ReLuGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ReLuGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(3, hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(3, hidden_size,
                                                   hidden_size))
        self.bias = nn.Parameter(torch.Tensor(3, hidden_size))

    def forward(self, input, hx=None):
        weight_ih = self.weight_ih
        bias = self.bias
        weight_hh = self.weight_hh

        z = (torch.mv(weight_ih[0], input) + torch.mv(weight_hh[0], hx) +
             bias[0]).sigmoid()
        r = (torch.mv(weight_ih[1], input) + torch.mv(weight_hh[1], hx) +
             bias[1]).sigmoid()
        n = (torch.mv(weight_ih[2], input) + torch.mv(weight_hh[2], hx * r) +
             bias[2]).relu()
        h = (torch.ones_like(z) - z) * n + z * hx

        return h


class AttnGRU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, atten_dim):
        super(AttnGRU, self).__init__()
        self.hidden_dim = hidden_dim

        self.attn = Attention(hidden_dim, atten_dim)
        self.xy_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True))

        self.hidden_embed = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.ReLU(inplace=True))

        self.gru = ReLuGRUCell(hidden_dim * 2, hidden_dim)
        self.predict_linear = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def forward(self, xy, temp_hidden, spat_hidden, prev_hidden):
        H = self.attn(temp_hidden, spat_hidden)

        x_e = self.xy_embed(xy)  # (input_dim,) -> (hidden_dim,)

        cat_H = torch.cat([temp_hidden, H])
        H_e = self.hidden_embed(cat_H)  # (hidden_dim * 2,) -> (hidden_dim,)

        cat = torch.cat([x_e, H_e])  # -> (hidden_dim * 2,)
        hidden = self.gru(
            cat,
            prev_hidden)  # (hidden_dim * 2,) (hidden_dim,) -> (hidden_dim,)
        predict = self.predict_linear(hidden)  # (hidden_dim,) -> (output_dim,)

        return [predict, hidden]

    def init_weights(self):
        optim_range = np.sqrt(1. / self.hidden_dim)

        self.xy_embed[0].weight.data.uniform_(-optim_range, optim_range)
        self.hidden_embed[0].weight.data.uniform_(-optim_range, optim_range)

        self.gru.weight_ih.data.uniform_(-optim_range, optim_range)
        self.gru.weight_hh.data.uniform_(-optim_range, optim_range)
        self.gru.bias.data.zero_()

        self.predict_linear.weight.data.uniform_(-optim_range, optim_range)
        self.predict_linear.bias.data.zero_()