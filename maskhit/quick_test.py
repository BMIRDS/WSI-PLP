import sys
import os
import glob
import re
from maskhit.trainer.fitter import HybridFitter
from maskhit.trainer.losses import FlexLoss
from options.train_options import TrainOptions
from utils.config import Config

study = sys.argv[1]
timestr = sys.argv[2]
timestr_new = sys.argv[3]

os.chdir(os.path.dirname(os.path.abspath(__file__)))
files = glob.glob(f'logs/{study}/{timestr}-*.log')
files = [x for x in files if not 'test' in x]
files.sort()

print("Log files found:")
for log_file in files:
    print(log_file)

for i, log_file in enumerate(files):
    with open(log_file, 'r') as f:
        org_cmd = f.readline().rstrip()

    org_cmd = org_cmd.replace("Argument all_arguments:", '').replace("'", "")
    if "num-patches" in " ".join(sys.argv[4:]):
        org_cmd = org_cmd.replace(" --sample-all", "")
    ckp = os.path.basename(log_file.replace('_meta.log', ''))
    
    new_cmd = ' '.join([
        'python train.py', org_cmd,
        f' --resume={ckp} --mode=test --test-type=test --resume-epoch=BEST --timestr={timestr_new}'
    ] + sys.argv[5:])

    
    print(f"executing following new_cmd: {new_cmd}")
    os.system(new_cmd)
