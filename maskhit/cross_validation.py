'''
Use this script to run 5 fold cross-validation
'''

import sys
import os
from maskhit.trainer.fitter import HybridFitter
from maskhit.trainer.losses import FlexLoss
from options.train_options import TrainOptions
from utils.config import Config

study = sys.argv[1]
timestr = sys.argv[2]
config_file = sys.argv[3]

# args_config = default_options()
print(f"config_file: {config_file}")
config = Config(default_config_file = config_file)

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# model training
def batch_train():
    for i in range(5):
        args = [
            'python train.py',
            '--user-config-file', f'{config_file}',
            '--outcome-type=classification',
            '--sample-patient',
            '-b=4',
            '--override-logs',
            f'--timestr={timestr}'
        ]
        
        new_cmd = ' '.join(args + [f'--fold={i} --study={study} --timestr={timestr}'])
        print(new_cmd)
        return_code = os.system(new_cmd)
        if return_code:
            return return_code
    return 0


if __name__ == '__main__':
    return_code = batch_train()
    if return_code:
        pass
    else:
        # model testing
        new_cmd = f'python quick_test.py {study} {timestr} {timestr}-test {config_file}'
        new_cmd += ' --prob-mask=0 --prop-mask=0,1,0 --mlm-loss=null'
        new_cmd = new_cmd.replace(' --override-logs', '')
        print(f"Creating this new_cmd in cross_validation.py {new_cmd}")
        os.system(new_cmd)
        print("FINISHED EXECUTING")
