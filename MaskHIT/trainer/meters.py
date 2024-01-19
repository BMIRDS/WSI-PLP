import numpy as np


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
            variance = np.average((np.array(self.values) - self.avg)**2,
                                  weights=self.weights)
            self.std = np.sqrt(variance)
        except Exception as e:
            print(e)
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

    def __init__(self,
                 num_batches,
                 meters,
                 verbose=True,
                 prefix="",
                 writer=None):
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