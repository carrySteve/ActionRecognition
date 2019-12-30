import sys
import numpy as np
import torch
import torch.nn as nn
import copy

from torch.nn import functional as F


class IndividualGraphModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IndividualGraphModule, self).__init__()
        self.theta = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        self.phi = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1)
        self.init_weights()

    def forward(self, x):
        """
        param x: (batch_size, feature_dim, time_step, person_num)
            [B, 2048, T, 12]
        return: individual graph C
        """
        individual_graph = self.embedded_gaussian(x)
        return individual_graph

    def embedded_gaussian(self, x):
        theta_x = self.theta(x).squeeze(dim=1)
        # [N, C, T, V] -> [N, T, V]
        theta_x = theta_x.permute(0, 2, 1).contiguous()
        # [N, T, V] -> [N, V, T]
        phi_x = self.phi(x).squeeze(dim=1)
        # [N, C, T, V] -> [N, T, V]
        f = torch.matmul(theta_x, phi_x)
        # [N, V, T] * [N, T, V] -> [N, V, V]

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

        self.individual_graph_module = IndividualGraphModule(
            in_channels, kernel_size[1])

        self.spatial_gcn_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size[1])

        self.temporal_gcn_conv = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(kernel_size[0], 1),
                      stride=(stride, 1),
                      padding=temporal_gcn_conv_padding),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

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
                nn.ReLU(inplace=True))

        self.init_weights()

    def forward(self, feature_embed):
        batch_size, embed_dim, time_step, person_num = feature_embed.shape
        feature_res = self.residual(feature_embed)
        # spatial GCN
        individual_graph = self.individual_graph_module(feature_embed)
        # [B, 2048, T, 12] -> [B, 12, 12]

        spatial_gcn_feature = self.spatial_gcn_conv(feature_embed).view(
            batch_size, -1, person_num)
        # [B, 2048, T, 12] -> [B, 2048 * T, 12]

        spatial_gcn_feature = torch.matmul(spatial_gcn_feature,
                                           individual_graph)
        # [B, 2048 * T, 12] * [B, 12, 12] -> [B, 2048 * T, 12]

        spatial_gcn_feature = spatial_gcn_feature.view(batch_size, embed_dim,
                                                       time_step, person_num)

        # temporal GCN
        st_gcn_feature = self.temporal_gcn_conv(spatial_gcn_feature)
        # [B, 2048, T, 12] -> [B, 2048, T, 12]

        feature_res_relu = st_gcn_feature + feature_res

        return feature_res_relu

    def init_weights(self):
        self.spatial_gcn_conv.weight.data.normal_(0., 0.02)
        self.spatial_gcn_conv.bias.data.zero_()

        self.temporal_gcn_conv[3].weight.data.normal_(0., 0.02)
        self.temporal_gcn_conv[3].bias.data.zero_()


class ClassifierB(nn.Module):
    def __init__(self,
                 feature_dim,
                 embed_dim=2048,
                 temporal_kernel_size=3,
                 spatial_kernel_size=1,
                 person_num=12,
                 action_dim=9,
                 activity_dim=8,
                 dropout_ratio=0.2,
                 group_pool='max'):
        super(ClassifierB, self).__init__()
        self.embed_layer = nn.Sequential(
            nn.Linear(feature_dim, embed_dim, bias=True),
            nn.ReLU(inplace=True), nn.Dropout(p=dropout_ratio))

        # self.data_bn = nn.BatchNorm1d(embed_dim * person_num)
        self.data_bn = nn.BatchNorm2d(embed_dim)

        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.st_gcn_networks = nn.ModuleList(
            (SpatialTemporalGCN(embed_dim, embed_dim, kernel_size, 1),
             SpatialTemporalGCN(embed_dim, embed_dim, kernel_size, 1)))

        self.predict_action = nn.Linear(embed_dim, action_dim, bias=True)
        if group_pool == 'max':
            self.predict_activity = nn.Sequential(
                nn.MaxPool2d((person_num, 1)),
                nn.Linear(embed_dim, activity_dim, bias=True),
            )
        elif group_pool == 'avg':
            self.predict_activity = nn.Sequential(
                nn.AvgPool2d((person_num, 1)),
                nn.Linear(embed_dim, activity_dim, bias=True),
            )
        else:
            print('wrong group pooling')
            sys.exit(0)
        self.init_weights()

    def forward(self, feature):
        # feature: [B, T, 12, 26400]

        feature_embed = self.embed_layer(feature)
        # [B, T, 12, 26400] -> [B, T, 12, 2048]
        batch_size, time_step, person_num, embed_dim = feature_embed.shape

        # feature_embed = feature_embed.permute(0, 3, 2, 1).contiguous()
        # feature_embed = feature_embed.view(batch_size, -1, time_step)
        # # [B, T, 12, 2048] -> [B, 2048, 12, T] -> [B, 2048 * 12, T]
        # feature_embed = self.data_bn(feature_embed)
        # feature_embed = feature_embed.view(batch_size, embed_dim,
        #                                          person_num, time_step)
        # feature_embed = feature_embed.permute(0, 1, 3, 2)
        feature_pred = feature_embed.permute(0, 3, 1, 2)
        # feature_embed = self.data_bn(feature_embed)

        for st_gcn in self.st_gcn_networks:
            feature_pred = st_gcn(feature_pred)
            # [B, T, 12, 2048]

        feature_pred = feature_pred.permute(0, 2, 3, 1)
        # [B, 2048, T, 12] -> [B, T, 12, 2048]

        action_logits = self.predict_action(feature_pred)
        # [B, T, 12, 2048] -> [B, T, 12, 9]
        activity_logits = self.predict_activity(feature_pred).squeeze(dim=2)
        # [B, T, 12, 2048] -> [B, T, 1, 2048] -> [B, T, 1, 8]
        return action_logits, activity_logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.embed_layer[0].weight)
        self.embed_layer[0].bias.data.zero_()

        nn.init.xavier_uniform_(self.predict_action.weight)
        self.predict_action.bias.data.zero_()

        nn.init.xavier_uniform_(self.predict_activity[1].weight)
        self.predict_activity[1].bias.data.zero_()
