import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange


def get_attention_map(attn_map):
    """Computes attention rollout results.
    Reference:
        https://arxiv.org/abs/2005.00928
    """

    # Average the attention weights across all heads.
    att_mat = torch.mean(attn_map, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1)).to(attn_map.device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(attn_map.device)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n],
                                           joint_attentions[n - 1])

    v = joint_attentions[-1]
    return v.detach()


def get_attention_map_multi(attn_maps):
    if attn_maps.size(1) == 0:
        raise ValueError("Attention map is empty")
    
    results = []
    for x in range(attn_maps.size(1)):
        attn_map = get_attention_map(attn_maps[:, x])
        results.append(attn_map)

    return torch.stack(results)



class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        attn = x['attn']
        # dots = x['dots']
        x = x['out']
        output = self.fn(x, **kwargs)
        out = output['out']
        if output['attn'] is None or attn is None:
            pass
        else:
            attn.append(output['attn'].detach().data)
        return {'out': out + x, 'attn': attn, 'dots': None}


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim,
                                           dim), nn.Dropout(dropout))

    def forward(self, x):
        return {"out": self.net(x), "attn": None}


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 global_only=False,
                 reattn=False,
                 is_last=False,
                 visualization=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        if reattn:
            self.reattn_weights = nn.Parameter(torch.randn(heads, heads))
            self.reattn_norm = nn.Sequential(Rearrange('b h i j -> b i j h'),
                                             nn.LayerNorm(heads),
                                             Rearrange('b i j h -> b h i j'))
        else:
            self.reattn_weights = None
            self.reattn_norm = None

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout))

        self.is_last = is_last
        self.visualization = visualization

    def forward(self, x, mask=None):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None:
            dots = dots.masked_fill(mask > 0, -1e4)

        attn = dots.softmax(dim=-1)

        # re-attention
        if self.reattn_weights is None:
            pass
        else:
            attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
            attn = self.reattn_norm(attn)

        # aggregate and out
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # the original part
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if self.visualization:
            output = {"out": out, "attn": attn, "dots": dots}
        else:
            output = {"out": out, "attn": None, "dots": None}
        return output


class Transformer(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 dropout=0.,
                 reattn=False,
                 visualization=False,
                 args=None):
        super().__init__()
        self.visualization = visualization
        self.layers = nn.ModuleList([])
        for di in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Residual(
                        PreNorm(
                            dim,
                            Attention(dim,
                                      heads=heads,
                                      dim_head=dim_head,
                                      dropout=dropout,
                                      global_only=True,
                                      reattn=reattn,
                                      is_last=di == depth - 1,
                                      visualization=visualization))),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim,
                                                 dropout=dropout)))
                ]))

        self.args = args

    def forward(self, x, mask=None):
        x = {'out': x, 'attn': [], 'dots': []}
        for attn, ff in self.layers:
            output = attn(x, mask=mask)
            x = ff(output)


        if self.visualization:
            if self.args.vis_head is None and self.args.vis_layer is None:
                attn_map = get_attention_map_multi(
                    torch.stack(x['attn'])[:, :, :, :, :])
            elif self.args.vis_head is None and self.args.vis_layer is not None:
                attn_map = get_attention_map_multi(
                    torch.stack(x['attn'])[
                        self.args.vis_layer, :, :, :, :].cpu().unsqueeze(0))
            elif self.args.vis_layer is None and self.args.vis_head is not None:
                attn_map = get_attention_map_multi(
                    torch.stack(x['attn'])
                    [:, :, self.args.vis_head, :, :].cpu().unsqueeze(2))
            else:
                attn_map = get_attention_map_multi(
                    torch.stack(x['attn'])
                    [self.args.vis_layer, :,
                     self.args.vis_head, :, :].cpu().unsqueeze(0).unsqueeze(2))


            output = {
                'out': x['out'],
                'attn': attn_map.cpu(),
                'dots': None
            }

        else:
            output = {'out': x['out'], 'attn': None, 'dots': None}
        return output
