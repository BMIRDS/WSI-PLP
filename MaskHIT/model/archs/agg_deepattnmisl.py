import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class AggDeepAttnMISL(nn.Module):
    """
    customized implementation
    """

    def __init__(self,
                 in_dim,
                 h=8,
                 layers=1,
                 dropout=0.1,
                 window_size=(10, 10),
                 args=None):
        super().__init__()

        self.h = h
        self.args = args
        self.dim = in_dim
        # self.dim = 520
        self.dropout = nn.Dropout(p=dropout)
        self.d = 64 // h
        self.proj_pre = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU())
        self.attention = nn.Sequential(
            nn.Linear(64, 32),  # V
            nn.Tanh(),
            nn.Linear(32, 1)  # W
        )
        self.centroids = nn.Parameter(
            torch.tensor(torch.load(args.cluster_centers)))
        self.iter = 1
        self.dim = 32
        self.proj_out = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                      nn.Dropout(p=dropout))

    def forward(self, inputs, **args):
        x = inputs['x']
        dist = torch.cdist(x, self.centroids)  # nph
        # weights belongs to each cluster
        wc = torch.softmax(10000000 / dist, dim=-1)
        wc = (wc > 0.5).float()
        # cluster specific features
        wc = F.normalize(wc, p=1, dim=1)  # n p h
        mask = wc.sum(1) == 0  # n h
        # shrink
        x = self.proj_pre(x)  # np*64
        x_cls = torch.einsum('npd,nph->nhd', x, wc)
        # attention
        A = self.attention(x_cls).squeeze(-1)  # nh1
        A.masked_fill_(mask, -1e5)
        wa = A.softmax(dim=-1)
        # combine
        x_cls = torch.einsum('nhd,nh->nd', x_cls, wa)
        out = self.proj_out(x_cls)

        enc_seq = org_seq = rearrange(x, 'b p d -> (b p) d')
        outputs = {'enc_cls': out, 'enc_seq': enc_seq, 'org_seq': org_seq}
        return outputs
