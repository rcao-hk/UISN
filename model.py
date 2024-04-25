""" USIN model definition.
    Author: Rui Cao
"""

import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F
from models.resunet14 import MinkUNet14D


class SuctionNet(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.feature_dim, D=3)
        self.suction_scoring = SuctionScoringNet(feature_dim=self.feature_dim)

    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']  # use all sampled point cloud, B*Ns*3
        B, point_num, _ = seed_xyz.shape  # batch _size

        # point-wise features
        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)
        end_points = self.suction_scoring(seed_features, end_points)
        return end_points


class SuctionScoringNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.in_dim = feature_dim
        self.conv_scoring = nn.Conv1d(self.in_dim, 1, 1)

    def forward(self, seed_features, end_points):
        suction_score = self.conv_scoring(seed_features)  # (B, 3, num_seed)
        end_points['score_pred'] = suction_score[:, 0]
        
        return end_points


class SuctionNet_prob(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.feature_dim, D=3)
        self.score_head = nn.Conv1d(self.feature_dim, 1, 1)
        self.sigma_head = nn.Conv1d(self.feature_dim, 1, 1)

    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']  # use all sampled point cloud, B*Ns*3
        B, point_num, _ = seed_xyz.shape  # batch _size

        # point-wise features
        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)
        score = self.score_head(seed_features)
        sigma = self.sigma_head(seed_features)
        sigma = F.softplus(sigma)
        end_points['score_pred'] = score.squeeze(1)
        end_points['sigma_pred'] = sigma.squeeze(1)
        return end_points