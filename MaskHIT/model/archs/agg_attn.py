import torch
import torch.nn as nn
from einops import rearrange

class AggAttn(nn.Module):
    """
    attention model in ilse 2018 paper: 
    https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
    """

    def __init__(self,
                 in_dim,
                 h=8,
                 layers=1,
                 dropout=0.,
                 window_size=(10, 10),
                 args=None):
        super().__init__()
        self.h = 1
        self.args = args
        self.attn = nn.Sequential(nn.Linear(in_dim, 512), nn.Tanh(),
                                  nn.Linear(512, self.h))
        self.dim = in_dim

    def forward(self, inputs, **args):
        x = inputs['x']
        nbatches, ppi = x.size(0), x.size(1)
        zeros = x.std(-1) == 0

        A = self.attn(x)  # N*P*C -> N*P*h

        # masking out zero paddings
        A = rearrange(A, 'n p h -> (n p) h')
        A[zeros.view(-1)] = -1e5
        A = rearrange(A, '(n p) h -> n p h', p=ppi)

        w = torch.softmax(A, dim=1)  # N*P*1

        enc_seq = org_seq = rearrange(x, 'b p d -> (b p) d')

        x = torch.bmm(w.permute(0, 2, 1), x.view(nbatches, ppi,
                                                 -1))  # N*1*P x N*P*C -> N*1*C
        outputs = {
            'enc_cls': x.squeeze(1),
            'enc_seq': enc_seq,
            'org_seq': org_seq
        }
        return outputs
