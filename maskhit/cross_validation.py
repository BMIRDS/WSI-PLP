'''
Use this script to run 5 fold cross-validation
CUDA_VISIBLE_DEVICES=5 python cross_validation.py ibd_project 2023_5_30 configs/config_ibd_train.yml


CUDA_VISIBLE_DEVICES=5 python cross_validation.py ibd_project 2023_5_30 configs/config_ibd_train.yml configs/config_default.yaml
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
config_file_default = sys.argv[4]

# args_config = default_options()
print(f"config_file: {config_file}")
config = Config(config_file_default, config_file)

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# model training
def batch_train():
    for i in range(1):
        args = [
            'python train.py',
            '--user-config-file', f'{config_file}',
            '--default-config-file', f'{config_file_default}',
            f'--timestr={timestr}'
        ]
        
        new_cmd = ' '.join(args + [f'--fold={i}'])
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
