import torch
import math
import torch.nn as nn
from einops import rearrange


class AggMHAttn(nn.Module):

    def __init__(self,
                 in_dim,
                 h=8,
                 layers=1,
                 dropout=0.1,
                 args=None,
                 model_name=None):
        super().__init__()

        self.h = h
        self.args = args
        self.model_name = model_name
        self.dim = in_dim
        self.dropout = nn.Dropout(p=dropout)
        self.d = in_dim // h

        self.proj_v = nn.Identity()

        self.proj_k = nn.Sequential(nn.Linear(in_dim, in_dim, bias=False),
                                    nn.Dropout(p=dropout), nn.ReLU())

        stdv_Q = 1. / math.sqrt(self.d)
        self.Q = nn.Parameter(
            torch.Tensor(self.h, self.d).uniform_(-stdv_Q, stdv_Q))

    def forward(self, inputs, **args):
        x = inputs['x']
        nbatches, ppi = x.size(0), x.size(1)
        zeros = x.std(-1) == 0

        v = self.proj_v(x).view(nbatches, ppi, self.h, -1)
        k = self.proj_k(x).view(nbatches, ppi, self.h, -1)
        A = torch.einsum('nphd,hd->nph', k, self.Q)

        # masking out zero paddings
        A = rearrange(A, 'n p h -> (n p) h')
        A[zeros.view(-1)] = -1e5
        A = rearrange(A, '(n p) h -> n p h', p=ppi)

        w = torch.softmax(A, dim=1)
        enc_cls = torch.einsum('nphd,nph->nhd', v, w).view(nbatches, -1)
        enc_seq = org_seq = rearrange(x, 'b p d -> (b p) d')

        outputs = {'enc_cls': enc_cls, 'enc_seq': enc_seq, 'org_seq': org_seq}
        return outputs
