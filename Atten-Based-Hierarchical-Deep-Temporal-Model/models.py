import sys
import numpy as np
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim, atten_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.atten_dim = atten_dim

        self.bn_hidden = nn.BatchNorm1d(hidden_dim)
        self.temporal_edge_layer = nn.Linear(hidden_dim, atten_dim)
        self.spatial_edge_layer = nn.Linear(hidden_dim, atten_dim)

    def forward(self, temp_hidden, spat_hidden):
        person_num = temp_hidden.shape[0]
        num_edges = spat_hidden.shape[-1]

        # Embed the temporal edgeRNN hidden state
        temporal_embed = self.temporal_edge_layer(
            self.bn_hidden(temp_hidden)).view(person_num, self.atten_dim, 1)
        # (person_num, hidden_dim,) -> (person_num, attn_dim, 1)

        # Embed the spatial edgeRNN hidden states
        spat_hidden = self.bn_hidden(spat_hidden).view(person_num, num_edges,
                                                       self.hidden_dim)
        spatial_embed = self.spatial_edge_layer(spat_hidden)
        # (person_num, hidden_dim, num_edges) -> (person_num, num_edges, attn_dim)

        # Dot based attention
        if num_edges > 1:
            # need mask
            attn = torch.matmul(spatial_embed, temporal_embed)
            # (person_num, num_edges, 1)
        else:
            # need change if not volleball dataset
            attn = torch.dot(spatial_embed.view(-1), temporal_embed)  # (1,)

        # Variable length
        # need different num_edges
        temperature = num_edges / np.sqrt(self.atten_dim)
        attn = torch.mul(attn, temperature)

        # Softmax
        weighted_value = nn.functional.softmax(attn, dim=0)

        # Compute weighted value
        # H = torch.mv(torch.t(spat_hidden), weighted_value.view(-1))
        H = torch.matmul(torch.transpose(spat_hidden, 1, 2),
                         weighted_value).view(person_num, self.hidden_dim)
        # (num_edges, hidden_dim) (num_edges,) ->  (hidden_dim,)
        # (person_num, num_edges, hidden_dim) (person_num, num_edges, 1) -> (person_num, hidden_dim)

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
        hiddens = []

        for i, h in zip(input, hx):
            z = (torch.mv(weight_ih[0], i) + torch.mv(weight_hh[0], h) +
                 bias[0]).sigmoid()
            r = (torch.mv(weight_ih[1], i) + torch.mv(weight_hh[1], h) +
                 bias[1]).sigmoid()
            n = (torch.mv(weight_ih[2], i) + torch.mv(weight_hh[2], h * r) +
                 bias[2]).relu()
            hiddens.append((torch.ones_like(z) - z) * n + z * h)

        h = torch.stack(hiddens)
        return h


class TanhGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TanhGRUCell, self).__init__()
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

        hiddens = []

        for i, h in zip(input, hx):
            z = (torch.mv(weight_ih[0], i) + torch.mv(weight_hh[0], h) +
                 bias[0]).sigmoid()
            r = (torch.mv(weight_ih[1], i) + torch.mv(weight_hh[1], h) +
                 bias[1]).sigmoid()
            n = (torch.mv(weight_ih[2], i) + torch.mv(weight_hh[2], h * r) +
                 bias[2]).tanh()
            hiddens.append((torch.ones_like(z) - z) * n + z * h)

        h = torch.stack(hiddens)
        return h


class ReactiveGRU(nn.Module):
    def __init__(self, input_dim, pout_dim, gout_dim, hidden_dim, atten_dim,
                 gru_activate, dropout):
        super(ReactiveGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_activate = gru_activate

        self.attn = Attention(hidden_dim, atten_dim)
        self.bn_feature = nn.BatchNorm1d(input_dim)
        self.bn_hidden1 = nn.BatchNorm1d(hidden_dim)
        self.bn_hidden2 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        self.feature_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True))

        self.hidden_embed = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.ReLU(inplace=True))

        if (gru_activate == "relu"):
            self.gru = ReLuGRUCell(hidden_dim * 2, hidden_dim)
        elif (gru_activate == "tanh"):
            self.gru = TanhGRUCell(hidden_dim * 2, hidden_dim)
        else:
            self.gru = nn.GRUCell(hidden_dim * 2, hidden_dim)
        # predict person actions
        self.predict_pact = nn.Sequential(self.dropout,
                                          nn.Linear(hidden_dim, pout_dim))
        # predict group activities
        self.predict_gact = nn.Sequential(self.dropout,
                                          nn.Linear(hidden_dim, gout_dim))

        self.init_weights()

    def forward(self, feature, temp_hidden, spat_hidden, prev_hidden):
        H = self.attn(temp_hidden, spat_hidden)

        # print(feature.shape)
        # print(self.bn_feature(feature).shape)
        feature_e = self.feature_embed(self.bn_feature(feature))
        # (person_num, input_dim) -> (person_num, hidden_dim)

        cat_H = torch.cat([temp_hidden, H], dim=-1)
        # -> (person_num, hidden_dim * 2)
        H_e = self.hidden_embed(self.bn_hidden2(cat_H))
        # (person_num, hidden_dim * 2) -> (person_num, hidden_dim)

        # print(feature_e.shape, H_e.shape)
        cat = torch.cat([feature_e, H_e], dim=-1)
        # -> (person_num, hidden_dim * 2)
        hidden = self.gru(cat, prev_hidden)
        # (person_num, hidden_dim * 2) (person_num, hidden_dim) -> (person_num, hidden_dim)
        person_act = self.predict_pact(self.bn_hidden1(hidden))
        # print('person_act', person_act.shape)
        # (person_num, hidden_dim) -> (person_num, pout_dim)
        group_act = self.predict_gact(self.bn_hidden1(hidden))
        # (person_num, hidden_dim) -> (person_num, gout_dim)
        # print('group_act', group_act.shape)

        return person_act, group_act, hidden

    def init_weights(self):
        optim_range = np.sqrt(1. / self.hidden_dim)

        self.feature_embed[0].weight.data.uniform_(-optim_range, optim_range)
        self.hidden_embed[0].weight.data.uniform_(-optim_range, optim_range)

        self.gru.weight_ih.data.uniform_(-optim_range, optim_range)
        self.gru.weight_hh.data.uniform_(-optim_range, optim_range)
        if (self.gru_activate == "relu" or self.gru_activate == "tanh"):
            self.gru.bias.data.zero_()
        else:
            self.gru.bias_ih.data.zero_()
            self.gru.bias_hh.data.zero_()

        self.predict_pact[1].weight.data.uniform_(-optim_range, optim_range)
        self.predict_pact[1].bias.data.zero_()

        self.predict_gact[1].weight.data.uniform_(-optim_range, optim_range)
        self.predict_gact[1].bias.data.zero_()
