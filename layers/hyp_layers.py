"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter, scatter_add
from layers.att_layers import DenseAtt, PygAtt, PygAttPlus, PygAttPlusPlus


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.hyp_linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.hyp_linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg, att_heads=1):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg, att_heads)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg, att_heads=1):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        self.edge_index = None
        self.edge_weight = None

        if self.use_att == 1:  # utilize dense attention
            self.att = DenseAtt(in_features, dropout)

        elif self.use_att == 2:  # sparse attention to deal with large dataset
            self.att = PygAtt(manifold, in_features, dropout, c, heads=att_heads)

        elif self.use_att == 3:
            self.att = PygAttPlus(manifold, in_features, dropout, c, heads=att_heads)

        elif self.use_att == 4:
            self.att = PygAttPlusPlus(manifold, in_features, dropout, c, heads=att_heads)

        else:
            self.att = None

    def norm(self, edge_index, num_nodes, edge_weight=None, dtype=None):
        if self.edge_index is not None:
            return self.edge_index, self.edge_weight

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        self.edge_index = edge_index
        self.edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return self.edge_index, self.edge_weight

    def agg_by_edge_index(self, x, edge_index):

        edge_index, edge_weight = self.norm(edge_index, x.size(0), None, x.dtype)
        edge_i = edge_index[0]
        edge_j = edge_index[1]
        x_j = torch.nn.functional.embedding(edge_j, x)  # Get the value of neighbor node x_j
        support = edge_weight.view(-1, 1) * x_j  # Calculate the weight of each neighbor j of node i
        result = scatter(src=support, index=edge_i, dim=0, dim_size=x.size(0), reduce='sum')
        return self.manifold.proj(self.manifold.expmap0(result, c=self.c), c=self.c)

    def forward(self, x, adj):
        if adj.size(0) != adj.size(1):  # adj here is edge_index
            return self.agg_by_edge_index(x, adj)

        x_tangent = self.manifold.logmap0(x, c=self.c)

        if self.use_att in [2, 3, 4]:  # sparse attention using torch-geometric
            support_t = self.att(x_tangent, adj)

        elif self.use_att == 1:  # dense attention
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)

        else:  # no attention
            support_t = torch.spmm(adj, x_tangent)

        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class LorentzGraphNeuralNetwork(nn.Module):
    def __init__(self, manifold, in_feature, out_features, c_in, c_out, drop_out, act, use_bias, use_att):
        super(LorentzGraphNeuralNetwork, self).__init__()
        self.c_in = c_in
        self.linear = LorentzLinear(manifold, in_feature, out_features, c_in, drop_out, use_bias)
        self.agg = LorentzAgg(manifold, c_in, use_att, out_features, drop_out)
        self.lorentz_act = LorentzAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x) ## problem is h1+
        h = self.agg.forward(h, adj)
        h = self.lorentz_act.forward(h)
        output = h, adj
        return output

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.agg.reset_parameters()

class LorentzLinear(nn.Module):
    # Lorentz Hyperbolic Graph Neural Layer
    def __init__(self, manifold, in_features, out_features, c, drop_out, use_bias):
        super(LorentzLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.drop_out = drop_out
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features-1))   # -1 when use mine mat-vec multiply
        self.weight = nn.Parameter(torch.Tensor(out_features-1, in_features-1))  # -1, 0 when use mine mat-vec multiply
        self.reset_parameters()

    def report_weight(self):
        print(self.weight)

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)
        # print('reset lorentz linear layer')

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.drop_out, training=self.training)
        mv = self.manifold.matvec_regular(drop_weight, x, self.bias, self.c, self.use_bias)
        return mv

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class LorentzAgg(Module):
    """
    Lorentz centroids aggregation layer
    """
    def __init__(self, manifold, c, use_att, in_features, dropout):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.use_att = use_att
        self.in_features = in_features
        self.dropout = dropout
        self.this_spmm = SpecialSpmm()
        if use_att:
            self.att = LorentzSparseSqDisAtt(manifold, c, in_features, dropout)


    def lorentz_centroid(self, weight, x, c):
        """
        Lorentz centroid
        :param weight: dense weight matrix. shape: [num_nodes, num_nodes]
        :param x: feature matrix [num_nodes, features]
        :return: the centroids of nodes [num_nodes, features]
        """
        if self.use_att:
            sum_x = self.this_spmm(weight[0], weight[1], weight[2], x)
        else:
            sum_x = torch.spmm(weight, x)
        x_inner = self.manifold.minkowski_dot(sum_x, sum_x, keepdim=False)
        coefficient = (c ** 0.5) / torch.sqrt(torch.abs(x_inner))
        return torch.mul(coefficient, sum_x.transpose(-2, -1)).transpose(-2, -1)

    def forward(self, x, adj):
        if self.use_att:
            adj = self.att(x, adj)
        output = self.lorentz_centroid(adj, x, self.c)
        return output

    def extra_repr(self):
        return 'c={}, use_att={}'.format(
                self.c, self.use_att
        )

    def reset_parameters(self):
        if self.use_att:
            self.att.reset_parameters()
        # print('reset agg finished')

class LorentzAct(Module):
    """
    Lorentz activation layer
    """
    def __init__(self, manifold, c_in, c_out, act):
        super(LorentzAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, self.c_in)
        return self.manifold.expmap0(xt, c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
                self.c_in, self.c_out
        )

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        device = b.device
        a = torch.sparse_coo_tensor(indices, values, shape, device=device)
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

class LorentzSparseSqDisAtt(nn.Module):
    def __init__(self, manifold, c, in_features, dropout):
        super(LorentzSparseSqDisAtt, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.manifold = manifold
        self.c = c
        self.weight_linear = LorentzLinear(manifold, in_features, in_features+1, c, dropout, True)

    def forward(self, x, adj):
        d = x.size(1) - 1
        x = self.weight_linear(x)
        index = adj._indices()
        _x = x[index[0, :]]
        _y = x[index[1, :]]
        _x_head = _x.narrow(1, 0, 1)
        _y_head = _y.narrow(1, 0, 1)
        _x_tail = _x.narrow(1, 1, d)
        _y_tail = _y.narrow(1, 1, d)
        l_inner = -_x_head.mul(_y_head).sum(-1) + _x_tail.mul(_y_tail).sum(-1)
        res = torch.clamp(-(self.c+l_inner), min=1e-10, max=1)
        res = torch.exp(-res)
        return (index, res, adj.size())
