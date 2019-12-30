import sys
import numpy as np
import torch
import torch.nn as nn
import copy

from torch.nn import functional as F


class IndividualGraphModuleGeneral(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IndividualGraphModuleGeneral, self).__init__()
        print('using general attention')
        self.theta = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        self.phi = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1)
        self.general_weights = nn.Parameter(
            torch.zeros(out_channels, out_channels))
        self.init_weights()

    def forward(self, x):
        """
        param x: (batch_size, feature_dim, time_step, person_num)
            [B, 1024, T, 12]
        return: individual graph C
        """
        batch_size, feature_dim, _, _ = x.shape

        h_s = self.theta(x).view(batch_size, feature_dim, -1)
        # [B, 1024, T, 12] -> [B, 1024, T*12]

        h_t = self.phi(x).view(batch_size, feature_dim, -1)
        # [B, 1024, T, 12] -> [B, 1024, T*12]

        f = torch.matmul(self.general_weights, h_s)
        # [1024, 1024] * [B, 1024, T*12] -> [B, 1024, T*12]
        f = torch.matmul(h_t.permute(0, 2, 1), f)
        # [B, T*12, 1024] * [B, 1024, T*12] -> [B, T*12, T*12]
        C = F.softmax(f, dim=-2)

        return C

    def init_weights(self):
        self.theta.weight.data.zero_()
        self.theta.bias.data.zero_()

        self.phi.weight.data.zero_()
        self.phi.bias.data.zero_()


class SpatialTemporalGCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.,
                 residual=True):
        super(SpatialTemporalGCN, self).__init__()
        # kernel_size (temporal_kernel_size, spatial_kernel_size)
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        temporal_gcn_conv_padding = ((kernel_size[0] - 1) // 2, 0)

        self.individual_graph_module = IndividualGraphModuleGeneral(
            in_channels, out_channels)

        self.gcn_bn = nn.BatchNorm2d(out_channels)
        self.res_bn = nn.BatchNorm2d(out_channels)

        self.spatial_gcn_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size[1])

        self.temporal_gcn_conv = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.Tanh(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(kernel_size[0], 1),
                      stride=(stride, 1),
                      padding=temporal_gcn_conv_padding),
            nn.BatchNorm2d(out_channels), nn.Tanh())

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=(stride, 1)), nn.BatchNorm2d(out_channels),
                nn.Tanh())

        self.init_weights()

    def forward(self, feature_embed):
        # [B, 1024, T, 12]
        batch_size, embed_dim, time_step, person_num = feature_embed.shape
        feature_res = self.residual(feature_embed)
        # spatial GCN
        spatial_gcn_feature = self.spatial_gcn_conv(feature_embed).view(
            batch_size, embed_dim, -1)
        # [B, 1024, T, 12] -> [B, 1024, T*12]

        individual_graph = self.individual_graph_module(feature_embed)
        # [B, 1024, T, 12] -> [B, T*12(atten), T*12]

        spatial_gcn_feature = torch.matmul(spatial_gcn_feature,
                                           individual_graph)
        # [B, 1024, T*12] * [B, T*12(atten), T*12] -> [B, 1024, T*12]

        spatial_gcn_feature = spatial_gcn_feature.view(batch_size, embed_dim,
                                                       time_step, person_num)

        # temporal GCN
        st_gcn_feature = self.temporal_gcn_conv(spatial_gcn_feature)
        # [B, 1024, T, 12] -> [B, 1024, T, 12]

        # st_gcn_feature = self.gcn_bn(st_gcn_feature)
        # feature_res = self.res_bn(feature_res)
        # feature_res_relu = torch.cat((st_gcn_feature, feature_res), dim=1)

        feature_res_relu = st_gcn_feature + feature_res
        # [B, 2048, T, 12]

        feature_pred = feature_res_relu.permute(0, 2, 3, 1)
        # [B, 2048, T, 12] -> [B, T, 12, 2048]

        return feature_pred

    def init_weights(self):
        self.spatial_gcn_conv.weight.data.normal_(0., 0.02)
        self.spatial_gcn_conv.bias.data.zero_()

        self.temporal_gcn_conv[3].weight.data.normal_(0., 0.02)
        self.temporal_gcn_conv[3].bias.data.zero_()


class ClassifierB(nn.Module):
    def __init__(self,
                 feature_dim,
                 embed_dim=2048,
                 hidden_dim=1024,
                 temporal_kernel_size=3,
                 spatial_kernel_size=1,
                 person_num=12,
                 action_dim=9,
                 activity_dim=8,
                 dropout_ratio=0.2,
                 group_pool='max',
                 rnn='gru'):
        super(ClassifierB, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.person_num = person_num
        self.action_dim = action_dim
        self.activity_dim = activity_dim
        self.rnn = rnn

        self.embed_layer = nn.Sequential(
            nn.Linear(feature_dim, embed_dim, bias=True),
            nn.ReLU(inplace=True), nn.Dropout(p=dropout_ratio),
            nn.Linear(embed_dim, hidden_dim, bias=True), nn.Tanh())

        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.st_gcn_networks = SpatialTemporalGCN(hidden_dim, hidden_dim,
                                                  kernel_size, 1)

        if rnn == 'gru':
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        elif rnn == 'lstm':
            self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        else:
            print('wrong rnn usage')
            sys.exit(0)

        self.predict_action = nn.Linear(hidden_dim, action_dim, bias=True)
        if group_pool == 'max':
            self.predict_activity = nn.Sequential(
                nn.MaxPool2d((person_num, 1)),
                nn.Linear(hidden_dim, activity_dim, bias=True),
            )
        elif group_pool == 'avg':
            self.predict_activity = nn.Sequential(
                nn.AvgPool2d((person_num, 1)),
                nn.Linear(hidden_dim, activity_dim, bias=True),
            )
        else:
            print('wrong group pooling')
            sys.exit(0)
        self.init_weights()

    def forward(self, feature):
        # feature: [B, T, 12, 26400]
        feature_embed = self.embed_layer(feature)
        # [B, T, 12, 26400] -> [B, T, 12, 1024]
        batch_size, time_step, person_num, hidden_dim = feature_embed.shape

        # spatial GCN
        feature_embed = feature_embed.permute(0, 3, 1, 2)
        # [B, T, 12, 1024] -> [B, 1024, T, 12]
        feature_res_activate = self.st_gcn_networks(feature_embed)
        # [B, T, 12, 2048]

        total_hiddens = []
        for batch_idx in range(batch_size):
            batch_hiddens = []
            prev_hidden = torch.zeros(self.person_num, self.hidden_dim).cuda()
            if self.rnn == 'lstm':
                prev_cell = torch.zeros(self.person_num,
                                        self.hidden_dim).cuda()
            for fidx in range(time_step):
                feature_batch_frame = feature_res_activate[batch_idx][fidx]
                # [12, 1024]
                if self.rnn == 'gru':
                    hidden = self.gru(feature_batch_frame, prev_hidden)
                elif self.rnn == 'lstm':
                    # print('using lstm ...')
                    hidden, cell = self.lstm(feature_batch_frame,
                                             (prev_hidden, prev_cell))
                    prev_cell = cell
                # [12, 1024] -> [12, 1024]
                prev_hidden = hidden
                batch_hiddens.append(hidden)
            batch_hiddens = torch.stack(batch_hiddens)
            # [T, 12, 9]
            total_hiddens.append(batch_hiddens)

        total_hiddens = torch.stack(total_hiddens)
        # [B, T, N, 1024]

        action_logits = self.predict_action(total_hiddens)
        # [B, T, 12, 1024] -> [B, T, 12, 9]
        activity_logits = self.predict_activity(total_hiddens).squeeze(dim=2)
        # [B, T, 12, 1024] -> [B, T, 1, 1024] -> [B, T, 1, 8] -> [B, T, 8]
        return action_logits, activity_logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.embed_layer[0].weight)
        self.embed_layer[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.embed_layer[3].weight)
        self.embed_layer[3].bias.data.zero_()

        if self.rnn == 'gru':
            nn.init.xavier_uniform_(self.gru.weight_ih)
            nn.init.xavier_uniform_(self.gru.weight_hh)
            self.gru.bias_ih.data.zero_()
            self.gru.bias_hh.data.zero_()
        elif self.rnn == 'lstm':
            nn.init.xavier_uniform_(self.lstm.weight_ih)
            nn.init.xavier_uniform_(self.lstm.weight_hh)
            self.lstm.bias_ih.data.zero_()
            self.lstm.bias_hh.data.zero_()

        nn.init.xavier_uniform_(self.predict_action.weight)
        self.predict_action.bias.data.zero_()

        nn.init.xavier_uniform_(self.predict_activity[1].weight)
        self.predict_activity[1].bias.data.zero_()
