import os
import pickle
import time
from pathlib import Path

from torch.utils.data import DataLoader
import torch

from data.dataset import get_dataloader
from data.transform import get_transformation
from model.fitter import HybridFitter
from model.helper import compose_logging
from model.loss import FlexLoss
from model.models import HybridModel
from options.train_options import TrainOptions

### Read parameter configuration
"""
Data loading process allows two input flows:
- Table based
    > Use commandline args
- List based
    > Use yaml config file
TODO: should be merged later. Issues are algorithms are slightly different so
that's why we separate them, for now.

Currently, it checks 
    args.config is None
to see if the input is
Table-based (Shuai)or List-based (Diana)

"""

opt = TrainOptions()
opt.initialize()
args = opt.parse()
    
    
### Read data mean and standard deviation if specified
if args.data_stats_mean is not None and args.data_stats_std is not None:
    data_stats = {'mean': args.data_stats_mean, 'std': args.data_stats_std}
else:
    data_stats = {'mean': [0.5,0.5,0.5], 'std': [0.25,0.25,0.25]}

os.chdir(os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # data transformation
    transform = get_transformation(mean=data_stats['mean'], std=data_stats['std'])
    loader = get_dataloader(args, transform)
    
    ### Define name of the logs to pre-defined or current time
    if len(args.timestr):
        TIMESTR = args.timestr
    else:
        TIMESTR = time.strftime("%Y%m%d_%H%M%S")
    writer = compose_logging(TIMESTR)

    for arg, value in sorted(vars(args).items()):
        writer['meta'].info("Argument %s: %r", arg, value)
    
    # Specify output size (1 for regression/survival and pre-defined for classification)
    if args.outcome_type in ['survival','regression']:
        num_classes = 1
        class_weights = None
    elif args.outcome_type == 'classification':
        num_classes = args.num_classes 
        if args.class_weights is not None:
            class_weights = list(map(float, args.class_weights.split(',')))
        else:
            class_weights = None

    # initialize model        
    model = HybridModel(
        backbone=args.backbone,
        pretrain=args.pretrain,
        outcome_dim=num_classes,
        outcome_type=args.outcome_type,
        random_seed=args.random_seed,
        dropout=args.dropout,
        device=device
    )

    writer['meta'].info(model)
    
    # use FlexLoss
    criterion = FlexLoss(
        outcome_type=args.outcome_type,
        class_weights=class_weights,
        device=device)
    
    # initialize trainer
    hf = HybridFitter(
        model=model,
        writer=writer,
        dataloader = loader,
        checkpoint_to_resume=args.resume,
        timestr=TIMESTR,
        args=args,
        model_name=TIMESTR,
        loss_function=criterion
    )
    
    
    # fitting the model or evaluate
    if args.mode == 'test':
        pass # will implement later or in a separate script
#         info_str = hf.evaluate(df_test, epoch=0)
#         writer['meta'].info(info_str)

    elif args.mode == 'train':
        hf.fit(checkpoints_folder=Path(args.checkpoint_dir)/TIMESTR)
        
    else:
        print("This mode has not been implemented!")