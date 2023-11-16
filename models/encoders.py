"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act, SAGEConvolution, SGCConvolution
# import utils.math_utils as pmath
from geoopt.manifolds.stereographic import PoincareBall
from geoopt.manifolds.lorentz import Lorentz


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c
        self.name = 'Encoder'

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output


class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        self.name = 'MLP'
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False


class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        self.name = 'GCN'
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True

class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
        self.name = 'GAT'
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True

class SGC(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(SGC, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        self.name = 'SGC'
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(SGCConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True


class SAGE(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(SAGE, self).__init__(c)
        assert args.num_layers > 0
        self.name = 'SAGE'
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(SAGEConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True



class HNN(Encoder):
    """
    Hyperbolic Neural Networks
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.name = 'HNN'
        self.manifold = getattr(manifolds, args.manifold)()
        # assert args.num_layers > 1
        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                hyp_layers.HNNLayer(
                    self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias)
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False  # In HNN, we do not encode the graph
        self.hyp_ireg = args.hyp_ireg  # flag for hyperbolic ireg

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        embeddings = super(HNN, self).encode(x_hyp, adj)

        if self.hyp_ireg in ['hire_tangent']:   # hire: whole replacement
            embeddings_tan = self.manifold.logmap0(embeddings, self.c)
            embeddings_tan = embeddings_tan - embeddings_tan.mean(dim=0)
            embeddings = self.manifold.proj(self.manifold.expmap0(embeddings_tan, self.c), self.c)
        return embeddings

class HGCN(Encoder):
    """
    Hyperbolic Graph Convolutional Networks
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        # assert args.num_layers > 1
        self.name = 'HGCN'
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HyperbolicGraphConvolution(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,
                    args.local_agg, args.n_heads
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.hyp_ireg = args.hyp_ireg  # flag for hyperbolic ireg

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        embeddings = super(HGCN, self).encode(x_hyp, adj)

        if self.hyp_ireg in ['hire_tangent', 'hire_tangent_centering_only']:   # hire: whole replacement
            embeddings_tan = self.manifold.logmap0(embeddings, self.c)
            embeddings_tan = embeddings_tan - embeddings_tan.mean(dim=0)
            embeddings = self.manifold.proj(self.manifold.expmap0(embeddings_tan, self.c), self.c)

        return embeddings


class LGCN(Encoder):
    """
    Lorentz Graph Convolutional Networks
    """

    def __init__(self, c, args):
        super(LGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        # assert args.num_layers > 1
        self.name = 'LGCN'
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        lgnn_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i] + 1, dims[i + 1] + 1
            # in_dim = in_dim + 1 if i != 0 else in_dim   # for layer more than 2
            act = acts[i]
            lgnn_layers.append(
                hyp_layers.LorentzGraphNeuralNetwork(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att
                )
            )
        self.layers = nn.Sequential(*lgnn_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_loren = self.manifold.proj(x, c=self.curvatures[0])
        return super(LGCN, self).encode(x_loren, adj)

    def reset_parameters(self):
        for tmp_layer in self.layers:
            tmp_layer.reset_parameters()


class HGNN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HGNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        # assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        self.name = 'HGNN'
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HGNNLayer(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,
                    args.local_agg
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGNN, self).encode(x_hyp, adj)





class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        self.name = 'Shallow'
        self.dr = args.dr
        if args.manifold == 'Hyperboloid':
            args.dim = args.dim + 1
        weights = torch.Tensor(args.n_nodes, args.dim).cuda()
        if not args.pretrained_embeddings:
            # weights = self.manifold.init_weights(weights, self.c)
            weights = self.manifold.proj(self.manifold.expmap0(self.manifold.init_weights(weights, self.c), self.c),
                                         self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)

        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if self.use_feats:
            args.dim = args.feat_dim + weights.shape[1]
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
            h = self.manifold.proj(h, self.c)

        if self.hyp_ireg in ['hire_tangent']:
            embeddings_tan = self.manifold.logmap0(h, self.c)
            embeddings_tan = embeddings_tan - embeddings_tan.mean(dim=0)
            h = self.manifold.proj(self.manifold.expmap0(embeddings_tan, self.c), self.c)

        return super(Shallow, self).encode(h, adj)
