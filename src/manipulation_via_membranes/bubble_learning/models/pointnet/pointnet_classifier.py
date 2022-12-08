"""
From: https://github.com/meder411/PointNet-PyTorch/blob/master/models/pointnet_classifier.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad

from manipulation_via_membranes.bubble_learning.models.pointnet.pointnet_base import PointNetBase


class PointNetClassifier(nn.Module):
    """
    Class for PointNetClassifier. Subclasses PyTorch's own "nn" module
    Computes the local embeddings and global features for an input set of points.
    """

    def __init__(self, K=3):
        super().__init__()

        # Local and global feature extractor for PointNet
        self.base = PointNetBase(K)

        # Classifier for ShapeNet
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 40))

    def forward(self, x):
        """
        Take as input a B x K x N matrix of B batches of N points with K dimensions
        :param x: (B, N, K) tensor
        :return:
        """

        # Only need to keep the global feature descriptors for classification
        # Output should be B x 1024
        x = x.transpose(-2,-1) # reshape to (B, K, N)
        x, _, T2 = self.base(x)
        out = self.classifier(x) # Returns a B x 40
        return out, T2


    @classmethod
    def get_name(cls):
        return 'pointnet_classifier'

    @property
    def name(self):
        return self.get_name()
