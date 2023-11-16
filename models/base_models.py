"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1
import geoopt.manifolds.lorentz.math as lmath


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        # if self.manifold.name == 'Hyperboloid':
        # args.feat_dim = args.feat_dim + 1
        # self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.args = args
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)
        self.d0 = []
        self.d2 = []
        self.hdo = []
        self.center = None
        self.activation = lambda x: x

    def get_c(self):
        # get the curvature of the output layer
        if self.encoder.name == 'HGNN':
            return self.encoder.layers[-1].c_in
        elif self.encoder.name in ['HGCN', 'HNN']:
            return self.encoder.layers[-1].hyp_act.c_out
        elif self.encoder.name == 'Shallow':
            return self.encoder.c
        elif self.encoder.name == 'LGCN':
            return self.encoder.layers[-1].lorentz_act.c_out
        else:
            raise Exception('please check the encoder name')

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def hir_loss(self, embeddings):
        c = self.get_c()
        # regularization on the tangent distance to the origin without changing original embeddings
        if self.args.hyp_ireg == 'hir_tangent':
            assert self.args.ireg_lambda != 0
            embeddings_tan = self.manifold.logmap0(embeddings, c)
            embeddings_tan = embeddings_tan - embeddings_tan.mean(dim=0)  # Equation (7)
            tangent_mean_norm = (1e-6 + embeddings_tan.pow(2).sum(dim=1).mean())
            tangent_mean_norm = self.activation(-tangent_mean_norm)
            return tangent_mean_norm

        # regularization on the tangent distance to the origin with changing original embeddings
        elif self.args.hyp_ireg == 'hire_tangent':
            assert self.args.ireg_lambda != 0
            embeddings_tan = self.manifold.logmap0(embeddings, c)
            # centering has been achieved before
            tangent_mean_norm = (1e-6 + embeddings_tan.pow(2).sum(dim=1).mean())
            tangent_mean_norm = self.activation(-tangent_mean_norm)
            return tangent_mean_norm

        # regularization on the tangent distance to the origin with change original embeddings
        elif self.args.hyp_ireg in ['hir_tangent_stretching_only']:
            assert self.args.ireg_lambda != 0
            embeddings_tan = self.manifold.logmap0(embeddings, c)
            tangent_mean_norm = (1e-6 + embeddings_tan.pow(2).sum(dim=1).mean())
            tangent_mean_norm = self.activation(-tangent_mean_norm)
            return tangent_mean_norm
        else:
            return 0

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)

        if self.args.hyp_ireg:  # self.hyp_ireg is not None
            loss_hir = self.hir_loss(embeddings)
            # loss = loss + self.args.ireg_lambda * loss_hir
            loss = loss + self.args.ireg_lambda * (max(loss_hir, -10) + 10) # more stable
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]