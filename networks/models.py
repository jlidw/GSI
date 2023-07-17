import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .layers import AdaptiveGraphLayer, GraphConvolution


# -----------------  GSI: with Adaptive Graph Structure Learning ----------------- #
class GSI(nn.Module):
    """
        This implementation is modified from GAT, refer to https://github.com/Diego999/pyGAT/blob/master/models.py
    """
    def __init__(self, nfeat, nhid, dropout, nheads, nnodes, nclass=1,
                 mask_act=None, adj_norm=None, sym_mask=True, mask_dropout=False):
        super(GSI, self).__init__()
        self.dropout = dropout

        self.adaptive_heads = [AdaptiveGraphLayer(nfeat, nhid, dropout=dropout, num_nodes=nnodes, concat=True,
                                                  mask_act=mask_act, adj_norm=adj_norm, sym_mask=sym_mask, mask_dropout=mask_dropout) for _ in range(nheads)]
        for i, head in enumerate(self.adaptive_heads):
            self.add_module('head_{}'.format(i), head)

        self.out_layer = AdaptiveGraphLayer(nhid * nheads, nclass, dropout=dropout, num_nodes=nnodes, concat=False,
                                            mask_act=mask_act, adj_norm=adj_norm, sym_mask=sym_mask, mask_dropout=mask_dropout)

    def forward(self, x, adj_mat1, adj_mat2):
        """
        Args:
            x: input
            adj_mat1: adjacency matrix for the 1st layer, without self-loop;
            adj_mat2: adjacency matrix for the 2nd layer, with self-loop;
        """
        x = torch.cat([head(x, adj_mat1) for head in self.adaptive_heads], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_layer(x, adj_mat2))

        return x.flatten()


# -----------------  Vanilla GCN: without Adaptive Graph Structure Learning ----------------- #
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, nclass=1):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    # two layers
    def forward(self, x, adj, adj_I):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj_I)

        return x.flatten()

