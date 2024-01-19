'''
This script is called by the `cross_validation.py` script.
By parsing log files, it runs the train script in test mode
resuming a model checkpoint that was trained either by
a separate invocation of the train script or from the batch_train
function in the `cross_validation.py` script.
'''

import sys
import os
import glob
from utils.config import Config
from pathlib import Path
import re

study = sys.argv[1]
timestr = sys.argv[2]
timestr_new = sys.argv[3]

# the first three arguments passed from `cross_validation.py`
print(f"{study}, {timestr}, {timestr_new}")

current_directory = Path(__file__).parent
os.chdir(current_directory)
files = glob.glob(f'logs/{study}/{timestr}-*.log')
files = [x for x in files if not 'test' in x]
files.sort()

print("Log files found:")
for log_file in files:
    print(log_file)

for i, log_file in enumerate(files):
    log_file_path = Path(log_file)
    with open(log_file, 'r') as f:
        org_cmd = f.readline().rstrip() # getting original arguments from meta file

    org_cmd = org_cmd.replace("Argument all_arguments:", '').replace("'", "")
    if "num-patches" in " ".join(sys.argv[4:]):
        org_cmd = org_cmd.replace(" --sample-all", "")
    ckp = log_file_path.stem.replace('_meta', '')

    # filter out the original timestr
    pattern = r'--timestr=[^\s]+'
    org_cmd = re.sub(pattern, '', org_cmd)

    timestr_new += '-test'
    new_cmd = ' '.join([
        'python train.py', org_cmd,
        f' --mode=test --test-type=test --resume-epoch=BEST --timestr={timestr_new}'
    ])

    print(f"executing following new_cmd: {new_cmd}")
    os.system(new_cmd)
