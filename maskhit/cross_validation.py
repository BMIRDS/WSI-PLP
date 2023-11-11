'''
Use this script to run cross-validation

Example Usage: CUDA_VISIBLE_DEVICES=8 python cross_validation.py --user-config-file "configs/config_ibd_train.yml" --default-config-file "configs/config_default.yaml" --folds 0
'''

import sys
import os
from maskhit.trainer.fitter import HybridFitter
from maskhit.trainer.losses import FlexLoss
from options.train_options import TrainOptions
from utils.config import Config

opt = TrainOptions()
opt.initialize()

opt.parser.add_argument(
        "--default-config-file", 
        type=str,
        default='configs/config_default.yaml',
        help="Path to the base configuration file. Defaults to 'config.yaml'.")
opt.parser.add_argument(
        "--user-config-file", 
        type=str,
        help="Path to the user-defined configuration file.")
opt.parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Optional list of folds for cross-validation.")

args = opt.parse()

config_file = args.user_config_file
config_file_default = args.default_config_file

# args_config = default_options()
print(f"config_file: {config_file}")
config = Config(config_file_default, config_file)
folds = [int(i) for i in args.folds.split(',')] if args.folds else list(range(config.dataset.num_folds))
print(f"Testing on folds: {list(folds)}")
study = config.dataset.study
timestr = config.dataset.timestr_model

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# model training
def batch_train():
    args_global = opt.parse()
    for i in folds:
        args = [
            'python train.py',
            '--user-config-file', f'{config_file}',
            '--default-config-file', f'{config_file_default}',
            f'--timestr={timestr}', f'--mil1={args_global.mil1}'
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
