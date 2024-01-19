import torch
from torch import nn

###########################################
#             Loss function               #
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

    scores = torch.clamp(scores, max=20)  # to avoid too large value after exponential
    scores_exp = torch.exp(scores)
    idx = torch.argsort(times, descending=True)
    times = times[idx]
    scores = scores[idx]
    scores_exp = scores_exp[idx]
    events = events[idx]
    log_scores = torch.log(torch.cumsum(scores_exp, dim=0) + 1e-5)
    uncensored_likelihood = scores - log_scores
    censored_likelihood = torch.mul(uncensored_likelihood, events.type(torch.float32)).sum()
    num_events = events.sum()
    loss = torch.div(-censored_likelihood, num_events)
    return loss


########################################
#          balanced sampling           #
########################################
# On the patient level, each epoch might be too small (several hundred patients).
# We can repeat all the samples multiple times and randomize them to obtain a larger
# epoch size.
# Specifically, for survival analysis, we can fixed the ratio of e (events) and ne 
# (no events) to ensure each batch contains some events.


class FlexLoss: ### To-do: change this in to nn.Module Class ?
    def __init__(self, outcome_type, class_weights=None, device=torch.device('cpu')):
        assert outcome_type in ['survival', 'classification', 'regression']
        if class_weights is None:
            pass
        else:
            class_weights = torch.tensor(class_weights).to(device)

        if outcome_type == 'survival':
            self.criterion = log_parlik_loss_cox
        elif outcome_type == 'classification':
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
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

        else:
            return self.criterion(pred, target.float())
