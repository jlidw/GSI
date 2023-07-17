import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, num_nodes,
                 mask_bias=False, concat=True, mask_act=None, adj_norm=None, sym_mask=True, mask_dropout=False):
        super(AdaptiveGraphLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.mask_bias = mask_bias
        self.mask_act = mask_act
        self.adj_norm = adj_norm
        self.sym_mask = sym_mask
        self.mask_dropout = mask_dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.mask, self.mask_bias = construct_edge_mask(num_nodes, init_strategy="const", is_bia=self.mask_bias)

    def get_masked_adj(self, adj):
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(mask)
        elif self.mask_act == "ELU":
            mask = nn.ELU()(mask)

        if self.sym_mask:
            mask = (mask + mask.t()) / 2

        masked_adj = adj * mask
        if self.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj

    def forward(self, input, adj):
        h = torch.mm(input, self.W)  # [N, out_features]
        masked_adj = self.get_masked_adj(adj)

        if self.adj_norm == "norm_1":
            masked_adj = F.normalize(masked_adj, p=1, dim=1)
        elif self.adj_norm == "norm_2":
            masked_adj = F.normalize(masked_adj, p=2, dim=1)
        elif self.adj_norm == "softmax":
            masked_adj = masked_adj.masked_fill(masked_adj == 0, -1e10)
            masked_adj = F.softmax(masked_adj, dim=1)

        if self.mask_dropout:
            masked_adj = F.dropout(masked_adj, self.dropout, training=self.training)

        h_prime = torch.matmul(masked_adj, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    """
    Vanilla GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def construct_edge_mask(num_nodes, init_strategy="normal", is_bia=True, const_val=1.0):
    """ This implementation is modified from GNNExplainer, refer to
    https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py """
    mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
    if init_strategy == "normal":
        std = nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (num_nodes + num_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
            # mask.clamp_(0.0, 1.0)
    elif init_strategy == "const":
        nn.init.constant_(mask, const_val)
    if is_bia:
        mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        nn.init.constant_(mask_bias, 0.0)
    else:
        mask_bias = None

    return mask, mask_bias

