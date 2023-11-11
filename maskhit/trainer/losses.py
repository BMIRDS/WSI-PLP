import torch
from torch import nn
from .info_nce_loss import info_nce
from einops import rearrange
import torch.nn.functional as F

###########################################
#             ContrasiveLoss              #
###########################################


class ContrasiveLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.loss_fn = dict_loss[args.mlm_loss](args=args)
        self.args = args

    def forward(self, outputs):

        ################
        # loss calculation
        device = outputs['enc_cls'].device
        attn_loss_seq, attn_loss_cls = self.loss_fn(outputs)

        if self.args.no_cls_loss:
            attn_loss_cls = torch.tensor(0.).to(device)
        if self.args.no_seq_loss:
            attn_loss_seq = torch.tensor(0.).to(device)

        return attn_loss_seq, attn_loss_cls


class LossSeqCompInfonceL2(nn.Module):
    '''
    additional part to predict if a patch is background
    '''

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        # self.norm_seq = nn.LayerNorm(args.hidden_dim)
        self.l2 = nn.PairwiseDistance(p=2)
        # self.is_background = nn.Linear(args.hidden_dim, 2)
        # self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs):
        enc_seq = outputs['enc_seq']
        org_seq = outputs['org_seq']
        masks = outputs['masks']
        zeros = outputs['zeros']

        sel = (masks > 0) * ~zeros

        # ###############
        # the info-nce loss for the sequence encoding
        attn_loss_seq_1 = info_nce(query=enc_seq[sel],
                                   positive_key=org_seq[sel],
                                   temperature=0.1,
                                   reduction='mean',
                                   negative_mode='paired',
                                   to_normalize=False,
                                   dist='l2')

        attn_loss_seq_2 = self.l2(enc_seq[sel], org_seq[sel]).mean()

        # is_background = self.is_background(enc_seq)
        # ce_loss = self.ce(is_background[masks>0], zeros.long()[masks>0])

        return attn_loss_seq_1, attn_loss_seq_2 * 2


class LossNull(nn.Module):

    def __init__(self, args=None):
        super().__init__()

    def forward(self, outputs):
        device = outputs['enc_cls'].device
        return torch.tensor(0.).to(device), torch.tensor(0.).to(device)


dict_loss = {
    'compseqil2': LossSeqCompInfonceL2,
    "null": LossNull,
}

###########################################
#             Survival Loss               #
###########################################


def log_parlik_loss_cox(scores, times=None, events=None):
    '''
    scores: values predicted by CNN model
    times: follow-up time
    events: 1 for event 0 for censor
    '''
    scores = scores.view(-1)
    times = times.view(-1)
    events = events.float().view(-1)

    scores = torch.clamp(scores,
                         max=20)  # to avoid too large value after exponential
    scores_exp = torch.exp(scores)
    idx = torch.argsort(times, descending=True)
    times = times[idx]
    scores = scores[idx]
    scores_exp = scores_exp[idx]
    events = events[idx]
    log_scores = torch.log(torch.cumsum(scores_exp, dim=0) + 1e-5)
    uncensored_likelihood = scores - log_scores
    censored_likelihood = torch.mul(uncensored_likelihood,
                                    events.type(torch.float32)).sum()
    num_events = events.sum()
    loss = torch.div(-censored_likelihood, num_events)
    return loss


class FlexLoss:

    def __init__(self, outcome_type, weight=None):
        from maskhit.train import device # avoids circular import
        assert outcome_type in [
            'survival', 'classification', 'regression', 'mlm'
        ]
        if outcome_type == 'survival':
            self.criterion = log_parlik_loss_cox
        elif outcome_type == 'classification' or outcome_type == 'mlm':
            if weight is not None and not weight.empty:
                weight_tensor = torch.tensor(weight).to(device).float()
                self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            else: 
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        self.outcome_type = outcome_type

    def calculate(self, pred, target):
        if self.outcome_type == 'survival':
            time = target[:, 0].float()
            event = target[:, 1].int()
            return self.criterion(pred, time, event)

        elif self.outcome_type == 'classification':
            return self.criterion(pred, target.long().view(-1))

        elif self.outcome_type == 'mlm':
            # predict the patientID using the CLS token
            # return self.criterion(pred, target.long().view(-1))
            return 0 * pred.sum()
        else:
            return self.criterion(pred, target.float())
