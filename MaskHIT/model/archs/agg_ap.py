import torch.nn as nn
from .utils import masked_average


class AggAvgPool(nn.Module):

    def __init__(self, in_dim, h=8, layers=1, dropout=0.1, args=None):
        super().__init__()
        self.args = args
        self.proj = nn.Identity()
        self.dim = in_dim

    def forward(self, inputs, **args):
        x = inputs['x']
        zeros = x.std(-1) == 0
        feature_masks = ~zeros
        out_x = masked_average(x, feature_masks.unsqueeze(-1))

        outputs = {
            'enc_cls': out_x,
            'enc_seq': out_x,
            'org_seq': out_x
        }
        return outputs
