import sys
import pandas as pd
import glob
import re

cancer = sys.argv[1]
timestr = sys.argv[2]
files = glob.glob(f'logs/{cancer}/{timestr}-*.csv')
print(files)
files.sort()
files = [x for x in files if not re.match('.*-test.*-.*', x)][:5]


def read_one(fname, min_epochs):
    NULL_RESULTS = None, None, None
    # read in the meta information
    res = []
    try:
        with open(fname, 'r') as f:
            for line in f.readlines():
                res.append({
                    x.split(':')[0]: x.split(':')[1].strip()
                    for x in line.strip().split('\t')
                })
    except Exception as e:
        print(e)
        print(fname)
        return NULL_RESULTS
    dt_i = pd.DataFrame(res)

    if 'epoch' not in dt_i.columns:
        return NULL_RESULTS
    if 'auc-2yr' in dt_i.columns:
        # the prognosis prediction task
        dt_i['auc-2yr'] = dt_i['auc-2yr'].astype(float)
        dt_i['auc-5yr'] = dt_i['auc-5yr'].astype(float)
        dt_i['c-index'] = dt_i['c-index'].astype(float)
    elif 'auc' in dt_i.columns:
        # the prognosis prediction task
        dt_i['auc'] = dt_i['auc'].astype(float)
        dt_i['f1'] = dt_i['f1'].astype(float)
    elif 'loss' in dt_i.columns:
        pass
    else:
        return NULL_RESULTS
    if dt_i['epoch'].astype('int').max() < min_epochs:
        return NULL_RESULTS

    dt_i['epoch'] = dt_i['epoch'].astype(float)
    dt_i['loss'] = dt_i['loss'].astype(float)

    return dt_i


res = []
for fold, fname in enumerate(files):
    print(fname)
    res_i = read_one(fname, 0)
    res_i['fold'] = fold
    res.append(res_i)

df = pd.concat(res)
dfs = df.loc[df['mode'] == 'val'].groupby(['fold']).max()
dfss1 = pd.DataFrame([dfs.mean().to_dict()])
dfss2 = pd.DataFrame([dfs.std().to_dict()])

dfss1.index = ['avg']
dfss2.index = ['std']

print(pd.concat([dfs, dfss1, dfss2]))