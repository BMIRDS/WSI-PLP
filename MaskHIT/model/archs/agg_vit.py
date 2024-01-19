import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np

from maskhit.model.archs.utils import fill_zeros, fill_mask, masked_average, MaskingGenerator
from maskhit.model.archs.vit.deepvit import Transformer


class AggViT(nn.Module):

    def __init__(self,
                 in_dim,
                 h=8,
                 layers=1,
                 dropout=0.,
                 window_size=(10, 10),
                 args=None):

        super().__init__()

        self.args = args
        num_pos = 512
        hidden_dim = args.hidden_dim
        self.mask_token = nn.Parameter(torch.randn(in_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim))

        self.pos_embedding_x = nn.Parameter(
            torch.randn(1, num_pos, hidden_dim // 2))
        self.pos_embedding_y = nn.Parameter(
            torch.randn(1, num_pos, hidden_dim // 2))
        if args.zero_padding:
            self.zero_padding = nn.Parameter(torch.randn(in_dim))

        self.norm_feat = nn.LayerNorm(in_dim)
        self.proj_feat = nn.Conv2d(in_dim,
                                   args.hidden_dim,
                                   kernel_size=(1, 1),
                                   stride=(1, 1))

        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(dim=hidden_dim,
                                       depth=layers,
                                       heads=h,
                                       dim_head=hidden_dim // h,
                                       mlp_dim=args.mlp_dim,
                                       dropout=dropout,
                                       visualization=args.visualization,
                                       args=self.args)

        self.margin = args.region_size // args.patch_size

        if args.mlm_loss in ['cluster', 'null']:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(nn.Identity())
        self.dim = hidden_dim
        self.mpg = MaskingGenerator(
            self.margin,
            num_masking_patches=int(args.prob_mask * self.margin *
                                    self.margin),
            max_num_patches=self.args.block_max,
            min_num_patches=self.args.block_min,
            prop_mask=self.args.prop_mask)

    def forward(self, inputs, avg_pool=False, prob_mask=0.15):
        x = inputs['x']
        pos = inputs['pos']

        # fill zero paddings
        zeros = x.std(-1) == 0
        if self.args.zero_padding:
            x = fill_zeros(x, zeros, self.zero_padding)

        b, p, _ = x.shape
        n = 1

        if self.args.prob_mask == 0:
            masks = torch.zeros(b, p).to(x.device)
        else:
            masks = torch.tensor(
                np.stack([
                    self.mpg()[pos[ii, :, 0].cpu(), pos[ii, :, 1].cpu()]
                    for ii in range(b)
                ])).view(b, -1).to(x.device)

        # the new positional encoding
        if self.args.prob_mask == 0:
            masked_x = x
        else:
            masked_x = fill_mask(x, masks, self.mask_token)

        masked_x = rearrange(masked_x, 'b (n p) d -> (b n) p d', n=n)

        # proj
        cls_tokens = repeat(self.cls_token, '() p d -> b p d', b=b * n)
        masked_x = torch.cat((cls_tokens, masked_x), dim=1)

        masked_x = self.norm_feat(masked_x)
        masked_x = rearrange(masked_x, 'b (h w) d -> b d h w', h=1)
        masked_x = self.proj_feat(masked_x)
        masked_x = rearrange(masked_x, 'b d h w -> b (h w) d')

        if self.args.disable_position:
            pass
        else:
            pos = torch.cat([torch.zeros(b, 1, 2).to(pos.device), pos + 1],
                            dim=1).long()
            pos_embedding_x = self.pos_embedding_x[:, pos[:, :, 0], :]
            pos_embedding_y = self.pos_embedding_y[:, pos[:, :, 1], :]
            pos_embedding = torch.cat([pos_embedding_x, pos_embedding_y],
                                      dim=-1).squeeze(0)
            masked_x = pos_embedding + masked_x

        # feed to transformer
        masked_x = self.dropout(masked_x)

        if self.args.zero_padding:
            zero_mask = None
        else:
            zero_mask = torch.cat([torch.zeros(b, 1).to(x.device), zeros],
                                  dim=1)
            zero_mask = zero_mask.unsqueeze(1).repeat(1, zero_mask.size(1),
                                                      1).unsqueeze(1)

        tr_output = self.transformer(masked_x, zero_mask)
        out_x = tr_output['out']

        ################
        # get the sequence encoding
        enc_seq = out_x[:, 1:, :]
        enc_seq = rearrange(enc_seq, '(b n) p d -> (b n p) d', n=1)

        # standardize features
        org_seq = self.proj(x)
        org_seq = rearrange(org_seq, '(b n) p d -> (b n p) d', n=1)

        if masks.sum() == 0:
            sel = torch.ones_like(masks).view(-1).bool()
        else:
            sel = masks.view(-1) > 0
        sel = sel * ~zeros.view(-1)

        ################
        # get the cls encoding
        if avg_pool:
            sel = rearrange(sel, '(b p n) -> b p n', b=b, n=1)
            enc_cls = masked_average(out_x[:, 1:, :], sel)
        else:
            enc_cls = out_x[:, 0, :]

        outputs = {
            'enc_cls': enc_cls,
            'enc_seq': enc_seq,
            'org_seq': org_seq,
            'pos': pos,
            'masks': masks.view(-1),
            'zeros': zeros.view(-1),
            'attn': tr_output['attn'],
            'dots': tr_output['dots']
        }

        return outputs
