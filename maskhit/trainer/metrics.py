import pandas as pd
import numpy as np
import torch
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from scipy.special import softmax


def c_index(times, scores, events):
    try:
        cindex = concordance_index(times, scores, events)
    except Exception as e:
        cindex = 0.5
    return cindex


def xyear_auc(preds, times, events, cutpoint=5):
    preds = preds.reshape(-1)
    times = times.reshape(-1)
    events = events.reshape(-1)
    ebt = np.zeros_like(times) + 1
    ebt[(times < cutpoint) & (events == 0)] = -1
    ebt[times >= cutpoint] = 0
    ind = ebt >= 0
    try:
        auc = roc_auc_score(ebt[ind], preds[ind])
    except ValueError:
        auc = 0.5
    return auc


def find_confident_instance(preds):
    return preds[preds.max(1).argmax()]


def calculate_metrics(ids, preds, targets, outcome_type='survival'):
    #TODO: Should support all the outcome_types, or should raise error if a provided type is not available.
    #Can we create a class with the logics and feed the object, instead of implementing the logic here.
    df = pd.DataFrame(np.concatenate([ids, preds, targets], axis=1))

    if outcome_type == 'survival':
        df.columns = ['id', 'time', 'event', 'pred']
        df = df.groupby('id').mean()
        c = c_index(df.time, -df.pred, df.event)
        auc_2yr = xyear_auc(df.pred.to_numpy(),
                            df.time.to_numpy(),
                            df.event.to_numpy(),
                            cutpoint=2)
        auc_5yr = xyear_auc(df.pred.to_numpy(),
                            df.time.to_numpy(),
                            df.event.to_numpy(),
                            cutpoint=5)

        res = {'c-index': c, 'auc-2yr': auc_2yr, 'auc-5yr': auc_5yr}
        return res

    elif outcome_type == 'classification':
        df = pd.DataFrame(
            np.concatenate([ids, targets, softmax(preds, axis=1)], axis=1))
        targets = df.iloc[:, :2].groupby(0).mean().to_numpy().astype(int)
        preds = df.groupby(0).apply(
            lambda x: find_confident_instance(x.to_numpy()[:, 2:]))
        preds = np.stack(preds.to_list())
        f1 = f1_score(targets, preds.argmax(axis=1), average='weighted')

        if len(np.unique(targets)) != preds.shape[1]:
            auc = 0.5
        else:
            try:
                if preds.shape[1] > 2:
                    # multi-class
                    auc = roc_auc_score(targets.reshape(-1),
                                        torch.softmax(torch.tensor(preds),
                                                      dim=1),
                                        multi_class='ovr')
                else:
                    # binary
                    auc = roc_auc_score(
                        targets.reshape(-1),
                        torch.softmax(torch.tensor(preds), dim=1)[:, 1])
            except Exception as e:
                print(e)
                auc = 0.5
        res = {'f1': f1, 'auc': auc}

        return res

    elif outcome_type == 'regression':
        df.columns = ['id', 'pred', 'target']
        grouped = df.groupby('id').mean()  # Assuming averaging is the desired approach

        mae = mean_absolute_error(grouped['target'], grouped['pred'])
        mse = mean_squared_error(grouped['target'], grouped['pred'])
        r2 = r2_score(grouped['target'], grouped['pred'])

        res = {'MAE': mae, 'MSE': mse, 'r2': r2}
        return res

    else:
        raise ValueError(f"Unsupported outcome type: {outcome_type}")



class ModelEvaluation(object):

    def __init__(self,
                 outcome_type='survival',
                 loss_function=None,
                 mode='train',
                 variables=['ids', 'preds', 'targets'],
                 device=torch.device('cpu'),
                 timestr=None):

        self.outcome_type = outcome_type
        self.criterion = loss_function
        self.mode = mode
        self.timestr = timestr
        self.variables = variables
        self.device = device
        self.reset()

    def reset(self):
        self.data = {}
        for var in self.variables:
            self.data[var] = None

    def update(self, batch):
        for k, v in batch.items():
            if self.data[k] is None:
                self.data[k] = v.data.cpu().numpy()
            else:
                self.data[k] = np.concatenate(
                    [self.data[k], v.data.cpu().numpy()])

    def evaluate(self):
        metrics = calculate_metrics(self.data['ids'],
                                    self.data['preds'],
                                    self.data['targets'],
                                    outcome_type=self.outcome_type)

        loss_epoch = self.criterion.calculate(
            torch.tensor(self.data['preds']).to(self.device),
            torch.tensor(self.data['targets']).to(self.device))

        metrics['loss'] = loss_epoch.item()

        return metrics

    def save(self, filename):
        values = []
        for k, v in self.data.items():
            values.append(v)
        df = pd.DataFrame(np.concatenate(values, 1))
        if filename is None:
            return df
        else:
            df.to_csv(filename)
