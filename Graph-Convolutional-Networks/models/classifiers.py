import sys
import numpy as np
import torch
import torch.nn as nn


class ClassifierA(nn.Module):
    def __init__(self,
                 feature_dim,
                 embed_dim=2048,
                 bbox_num=12,
                 action_dim=9,
                 activity_dim=8,
                 dropout_ratio=0.2):
        super(ClassifierA, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.activity_dim = activity_dim
        self.embed_layer = nn.Sequential(
            nn.Linear(feature_dim, embed_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_ratio)
        )
        self.predict_action = nn.Linear(embed_dim, action_dim, bias=True)

        self.predict_activity = nn.Sequential(
            nn.MaxPool2d((bbox_num, 1)),
            nn.Linear(embed_dim, activity_dim, bias=True),
        )
        self.init_weights()

    def forward(self, feature):
        # feature: [B, 12, 26400]
        feature_embed = self.embed_layer(feature)
        # [B, 12, 26400] -> [B, 12, 2048]
        action_logits = self.predict_action(feature_embed).view(
            -1, self.action_dim)
        # [B, 12, 2048] -> [B, 12, 9] -> [B*12, 9]
        activity_logits = self.predict_activity(feature_embed.unsqueeze(dim=1))
        # [B, 12, 2048] -> [B, 1, 12, 2048] -> [B, 1, 1, 2048] -> [B, 1, 1, 8]
        activity_logits = activity_logits.view(-1, self.activity_dim)
        # [B, 1, 1, 8] -> [B, 8]
        return action_logits, activity_logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.embed_layer[0].weight)
        self.embed_layer[0].bias.data.zero_()

        nn.init.xavier_uniform_(self.predict_action.weight)
        self.predict_action.bias.data.zero_()

        nn.init.xavier_uniform_(self.predict_activity[1].weight)
        self.predict_activity[1].bias.data.zero_()


class ClassifierB(nn.Module):
    def __init__(self,
                 feature_dim,
                 embed_dim=2048,
                 hidden_dim=1024,
                 bboxes_num=12,
                 action_dim=9,
                 activity_dim=8,
                 dropout_ratio=0.8):
        super(ClassifierB, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.activity_dim = activity_dim
        self.embed_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=True),
            nn.Tanh()
        )

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.predict_action = nn.Linear(hidden_dim, action_dim, bias=True)

        self.predict_activity = nn.Sequential(
            nn.MaxPool2d((bboxes_num, 1)),
            nn.Linear(hidden_dim, activity_dim, bias=True),
        )
        self.init_weights()

    def forward(self, feature):
        # feature: [B, T, 12, 1024]
        batch_size = feature.shape[0]
        time_step = feature.shape[1]
        person_num = feature.shape[2]

        feature_embed = self.embed_layer(feature)
        # [B, T, 12, 2048] -> [B, T, 12, 1024]

        total_hiddens = []
        total_action_logits = []
        for batch_idx in range(batch_size):
            batch_hiddens = []
            prev_hidden = torch.zeros(person_num, self.hidden_dim).to(self.device)
            for fidx in range(time_step):
                feature_batch_frame = feature_embed[batch_idx][fidx]
                # [12, 1024]
                hidden = self.gru(feature_batch_frame, prev_hidden)
                # [12, 1024] -> [12, 1024]
                prev_hidden = hidden
                action_logits = self.predict_action(hidden).view(
                    -1, self.action_dim)
                # [12, 1024] -> [12, 9]
                # predict action
                total_action_logits.append(action_logits)
                # store hiddens
                batch_hiddens.append(hidden)
            batch_hiddens = torch.stack(batch_hiddens)
            # [T, 12, 9]
            total_hiddens.append(batch_hiddens)

        total_action_logits = torch.cat(total_action_logits, dim=0)
        # [BTN, 9]
        total_hiddens = torch.stack(total_hiddens)
        # [B, T, N, 1024]
        activity_logits = self.predict_activity(total_hiddens).view(-1, self.activity_dim)
        # [B, T, N, 1024] -> [B, T, 1, 1024] -> [B, T, 1, 8] -> [BT, 8]

        return total_action_logits, activity_logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.embed_layer[0].weight)
        self.embed_layer[0].bias.data.zero_()
        
        nn.init.xavier_uniform_(self.gru.weight_ih)
        nn.init.xavier_uniform_(self.gru.weight_hh)
        self.gru.bias_ih.data.zero_()
        self.gru.bias_hh.data.zero_()

        nn.init.xavier_uniform_(self.predict_action.weight)
        self.predict_action.bias.data.zero_()

        nn.init.xavier_uniform_(self.predict_activity[1].weight)
        self.predict_activity[1].bias.data.zero_()
