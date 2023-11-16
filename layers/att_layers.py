"""Attention layers (some modules are copied from https://github.com/Diego999/pyGAT."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import softmax
import math
class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.in_features = in_features

    def forward (self, x, adj):
        n = x.size(0)
        # n x 1 x d
        x_left = torch.unsqueeze(x, 1)
        x_left = x_left.expand(-1, n, -1)
        # 1 x n x d
        x_right = torch.unsqueeze(x, 0)
        x_right = x_right.expand(n, -1, -1)

        x_cat = torch.cat((x_left, x_right), dim=2)
        att_adj = self.linear(x_cat).squeeze()
        att_adj = torch.sigmoid(att_adj)
        att_adj = torch.mul(adj.to_dense(), att_adj)
        return att_adj


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.act = activation

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj._indices()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), ones)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        return self.act(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, activation, alpha, nheads, concat):
        """Sparse version of GAT."""
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.output_dim = output_dim
        self.attentions = [SpGraphAttentionLayer(input_dim,
                                                 output_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 activation=activation) for _ in range(nheads)]
        self.concat = concat
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, input):
        x, adj = input
        x = F.dropout(x, self.dropout, training=self.training)
        if self.concat:
            h = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        else:
            h_cat = torch.cat([att(x, adj).view((-1, self.output_dim, 1)) for att in self.attentions], dim=2)
            h = torch.mean(h_cat, dim=2)
        h = F.dropout(h, self.dropout, training=self.training)
        return (h, adj)

class PygAtt(nn.Module):

    def __init__(self, manifold, out_features, dropout, c=None, heads=1, concat=True):
        super(PygAtt, self).__init__()
        self.manifold = manifold
        self.dropout = dropout
        self.negative_slope = 0.02
        self.out_channels = out_features // heads
        assert heads < out_features
        self.heads = heads
        self.c = c
        self.concat = concat
        self.linear = nn.Linear(2 * self.out_channels, 1, bias=False)
        nn.init.xavier_normal_(self.linear.weight, gain=1.414)  # no bias

    def forward(self, x_tangent0, edges):
        if edges.layout is torch.sparse_coo:  # here edges is adj mat
            edge_index = edges._indices()
        else:
            edge_index = edges

        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]
        x_i = torch.nn.functional.embedding(edge_index_i, x_tangent0)
        x_j = torch.nn.functional.embedding(edge_index_j, x_tangent0)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        alpha = self.linear(torch.cat((x_i, x_j), dim=2)).squeeze()
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(src=alpha, index=edge_index_i, num_nodes=x_i.size(0))
        alpha = F.dropout(alpha, self.dropout, training=self.training)  # cannot remove, very important
        support_t = scatter(x_j * alpha.view(-1, self.heads, 1), edge_index_i, dim=0, reduce='sum')

        if self.concat:
            support_t = support_t.view(-1, self.heads * self.out_channels)
        else:
            support_t = support_t.mean(dim=1)

        return support_t

    def __repr__(self):
        return self.__class__.__name__ + ' (in_features=' + str(self.heads) + '*' + str(self.out_channels) + ', out_features=' + str(self.out_channels*self.heads) + ')'


class PygAttPlus(nn.Module):
    def __init__(self, manifold, out_features, dropout, c=None, heads=1, concat=True):
        super(PygAttPlus, self).__init__()
        self.manifold = manifold
        self.dropout = dropout
        self.out_channels = out_features // heads
        assert heads < out_features
        self.heads = heads
        self.c = c
        self.concat = concat
        self.linear = nn.Linear(2 * self.out_channels, 1, bias=False)
        nn.init.xavier_normal_(self.linear.weight, gain=1.414)

    def forward(self, x_tangent0, edge_index):
        if edge_index.layout is torch.sparse_coo:  # here edges is adj mat
            weights = edge_index._values()
            edge_index = edge_index._indices()

        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]

        x_i = torch.nn.functional.embedding(edge_index_i, x_tangent0)
        x_j = torch.nn.functional.embedding(edge_index_j, x_tangent0)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        alpha = self.linear(torch.cat((x_i, x_j), dim=2)).squeeze()
        weights = weights if len(alpha.size()) == 1 else weights.unsqueeze(1)
        beta = weights*alpha.sigmoid()
        support_t = scatter(x_j * beta.view(-1, self.heads, 1), edge_index_i, dim=0, reduce='sum')
        if self.concat:
            support_t = support_t.view(-1, self.heads * self.out_channels)
        else:
            support_t = support_t.mean(dim=1)

        return support_t

    def __repr__(self):
        return self.__class__.__name__ + ' (in_features=' + str(self.heads) + '*' + str(self.out_channels) + ', out_features=' + str(self.out_channels*self.heads) + ')'


class PygAttPlusPlus(nn.Module):
    def __init__(self, manifold, out_features, dropout, c=None, heads=1, concat=True):
        super(PygAttPlusPlus, self).__init__()
        self.manifold = manifold
        self.dropout = dropout
        self.out_channels = out_features // heads
        self.negative_slope = 0.02
        assert heads < out_features
        self.heads = heads
        self.c = c
        self.concat = concat
        self.linear = nn.Linear(2 * self.out_channels, 1, bias=True)

    def forward(self, x_tangent0, edge_index):
        if edge_index.layout is torch.sparse_coo:
            weights = edge_index._values()
            edge_index = edge_index._indices()

        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]

        x_i = torch.nn.functional.embedding(edge_index_i, x_tangent0)
        x_j = torch.nn.functional.embedding(edge_index_j, x_tangent0)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        alpha = self.linear(torch.cat((x_i, x_j), dim=2)).squeeze()
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(src=alpha, index=edge_index_i, num_nodes=x_i.size(0))
        weights = weights if len(alpha.size()) == 1 else weights.unsqueeze(1)
        alpha = weights*alpha
        # alpha = F.dropout(alpha, self.dropout, training=self.training)
        support_t = scatter(x_j * alpha.view(-1, self.heads, 1), edge_index_i, dim=0, reduce='sum')

        if self.concat:
            support_t = support_t.view(-1, self.heads * self.out_channels)
        else:
            support_t = support_t.mean(dim=1)

        return support_t

    def __repr__(self):
        return self.__class__.__name__ + ' (in_features=' + str(self.heads) + '*' + str(self.out_channels) + ', out_features=' + str(self.out_channels*self.heads) + ')'
