import random
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from torchvision import transforms
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, f1_score, r2_score, accuracy_score
from PIL import Image
import logging
import time
from collections import defaultdict

########################################
# setup the logging
########################################

def setup_logger(name, log_file, file_mode, to_console=False):
    """
        https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings
        To setup as many loggers as you want
    """

    formatter = logging.Formatter('%(message)s')

    handler = logging.FileHandler(log_file, mode=file_mode)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    if to_console:
        logger.addHandler(logging.StreamHandler())

    return logger


def compose_logging(model_name):
    writer = {}
    writer["meta"] = setup_logger("meta", os.path.join(
        "logs", "%s_meta.log" % model_name), 'w', to_console=True)
    writer["data"] = setup_logger("data", os.path.join(
        "logs", "%s_data.csv" % model_name), 'w', to_console=True)
    return writer


###########################################
#         commonly used functions         #
###########################################


def get_filename_extensions(args):
    ext_data = '%s_mag-%s_size-%s' % (args.cancer, args.magnification, args.patch_size)
    ext_experiment = 'by-%s_seed-%s' % (args.stratify, args.random_seed)
    ext_split = '%s_by-%s_seed-%s_nest-%s%s' % (args.cancer, args.stratify,
                                                args.random_seed, args.outer_fold, args.inner_fold)
    return ext_data, ext_experiment, ext_split


########################################
#               Early stopping         #
########################################
# modified from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = 0
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        print("Bad epochs: %s; Patience: %s; Best value: %6.4f" %
              (self.num_bad_epochs, self.patience, self.best))

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100)

###########################################
#             MISC FUNCTIONS              #
###########################################


# instantiate the model
def create_model(num_layers, pretrain, num_classes):
    assert num_layers in [18, 34, 50, 101, 152]
    architecture = "resnet{}".format(num_layers)
    model_constructor = getattr(torchvision.models, architecture)
    model = model_constructor(num_classes=num_classes)

    if pretrain is True:
        print("Loading pretrained model!")
        pretrained = model_constructor(pretrained=True).state_dict()
        if num_classes != pretrained['fc.weight'].size(0):
            del pretrained['fc.weight'], pretrained['fc.bias']
        model.load_state_dict(pretrained, strict=False)
    return model


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MobileNetV2Updated(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV2Updated, self).__init__()
        self.model = torchvision.models.mobilenet.mobilenet_v2(pretrained=pretrained)
        self.model.classifier = Identity()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, 1000)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


###########################################
#   RESULTS COLLECTION AND EVALUATION     #
###########################################

########################################
# by patch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.values = []
        self.ns = []
        self.weights = []

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.values.append(val)
        self.ns.append(n)
        self.weights = [x / self.count for x in self.ns]
        self.avg = np.average(self.values, weights=self.weights)

    def get_avg(self):
        return self.avg

    def get_std(self):
        try:
            variance = np.average((np.array(self.values) - self.avg)**2, weights=self.weights)
            self.std = np.sqrt(variance)
        except:
            self.std = 0.0
        return self.std

    def final(self):
        std = self.get_std()
        avg = self.get_avg()
        return {'avg': avg, 'std': std}

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, verbose=True, prefix="", writer=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.verbose = verbose
        self.writer = writer

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.verbose:
            if self.writer is not None:
                self.writer.info('\t'.join(entries))
            else:
                print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

########################################
# all batches combined

def c_index(times, scores, events):
    try:
        cindex = concordance_index(times, scores, events)
    except Exception as e:
        print(e)
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

def calculate_metrics(preds, targets, outcome_type='survival'):

    if outcome_type == 'survival':
        times, events = targets[:, 0], targets[:, 1]
        c = c_index(times, -preds.reshape(-1), events)
        auc_2yr = xyear_auc(preds, times, events, cutpoint=2)
        auc_5yr = xyear_auc(preds, times, events, cutpoint=5)

        res = {
            'c-index': c,
            'auc-2yr': auc_2yr,
            'auc-5yr': auc_5yr
        }

    elif outcome_type == 'classification':
        f1 = f1_score(targets, preds.argmax(axis=1), average='weighted')
        acc = accuracy_score(targets, preds.argmax(axis=1))
        try:
            if preds.shape[1] > 2:
                # multi-class
                auc = roc_auc_score(targets.reshape(-1),
                                    torch.softmax(torch.tensor(preds), dim=1), multi_class='ovr')
            else:
                # binary
                auc = roc_auc_score(targets.reshape(-1),
                                    torch.softmax(torch.tensor(preds), dim=1)[:, 1])
        except ValueError:
            auc = 0.5
        res = {
            'f1': f1,
            'auc': auc,
            'accuracy': acc
        }


    elif outcome_type == 'regression':
        r2 = r2_score(targets.reshape(-1), preds.reshape(-1))
        res = {
            'r2': r2
        }

    return res


class ModelEvaluation(object):
    def __init__(
        self,
        outcome_type='survival',
        loss_function=None,
        mode='train',
        variables=['preds', 'targets', 'id'],
        device=torch.device('cpu'),
        timestr=None
    ):

        self.outcome_type = outcome_type
        self.criterion = loss_function
        self.mode = mode
        self.timestr = timestr
        self.variables = variables
        self.device = device
        self.reset()

    def reset(self):
        self.data = dict()
        for var in self.variables:
            self.data[var] = None

    def update(self, batch):
        for k, v in batch.items():
            if isinstance(v, tuple):
                if self.data[k] is None:
                    self.data[k] = list()
                self.data[k] += list(v)
            else:
                # Tensor objects
                if self.data[k] is None:
                    self.data[k] = v.data.cpu().numpy()
                else:
                    self.data[k] = np.concatenate([self.data[k], v.data.cpu().numpy()])

    def evaluate(self):
        metrics = calculate_metrics(
            self.data['preds'],
            self.data['targets'],
            outcome_type=self.outcome_type)

        loss_epoch = self.criterion.calculate(  #FIXME
            torch.tensor(self.data['preds']).to(self.device),
            torch.tensor(self.data['targets']).to(self.device))

        metrics['loss'] = loss_epoch.item()

        return metrics

    def save(self, filename):
        values = []
        for k,v in self.data.items():
            values.append(v)
        df = pd.concat(objs=[
                pd.DataFrame(values[0]),
                pd.DataFrame(values[1]),
                pd.DataFrame(values[2])], axis=1)
        df.to_csv(filename)



