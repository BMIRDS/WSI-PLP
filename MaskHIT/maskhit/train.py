"""
USAGE: python train.py --user-config-file <filename> --sample-patient --batch-size <value> --override-logs --timestr <value> --fold <value>

REQUIRED OPTIONS:
  --user-config-file <filename>    Specify the user configuration file
  --fold <value>                   Specify the fold number

EXAMPLES:
- For IBD project
python train.py --user-config-file configs/config_ibd_train.yml --default-config-file configs/config_default.yaml  --timestr=2023_5_30 --fold=0


To test
Apptainer> CUDA_VISIBLE_DEVICES=8 python train.py  --user-config-file configs/config_ibd_train.yml --default-config-file configs/config_default.yaml --fold=0  --resume=2023_5_30-0 --mode=test --test-type=test --resume-epoch=BEST --timestr=2023_5_30_new-test
"""

# standard libraries
import ast
import glob
import os
import socket
import sys
import time
from pathlib import Path

# 3rd party
import pandas as pd
import torch

# custom/own libraries
from maskhit.trainer.fitter import HybridFitter
from maskhit.trainer.losses import FlexLoss
from options.train_options import TrainOptions
from utils.config import Config

"""TODO: The issues of this script:
    - inconsistency: config vs args
    - non-modular scripting style
    - args without doc
    - passing args to HybridFitter. this is like 'import *', should be specific.
    - formatting of printing messages.

"""


# Defining a global variable to store available device
global device

# checking to see if GPU is available for use
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print("\n")
opt = TrainOptions()
opt.initialize()

# adding additional arguments to the parser (e.g. config files)
opt.parser.add_argument(
        "--default-config-file", 
        type=str,
        default='configs/config_default.yaml',
        help="Path to the base configuration file. Defaults to 'config.yaml'.")
opt.parser.add_argument(
        "--user-config-file", 
        type=str,
        help="Path to the user-defined configuration file.")

args = opt.parse()
config = Config(args.default_config_file, args.user_config_file)
# string holding command-line arguments joined with spaces
args.all_arguments = ' '.join(sys.argv[1:])
#TODO: this var not used

assert not args.sample_all, "the argument --sample-all is deprecated, use --num-patches=0 instead"
    
# print(f"args.cancer: {args.cancer}")
if args.cancer == '.':
    args.cancer = ""
#TODO: Why we need to care this case?

# setting weight decay values
if hasattr(config.model, 'wd_attn') and hasattr(config.model, 'wd_fuse') and hasattr(config.model, 'wd_loss'):
    args.wd_attn = config.model.wd_attn
    args.wd_fuse = config.model.wd_fuse
    args.wd_loss = config.model.wd_loss
    pass
else:
    if hasattr(config.model, 'wd'):
        args.wd_attn = config.patch.wd
        args.wd_fuse = config.patch.wd
        args.wd_pred = config.patch.wd
        args.wd_loss = config.patch.wd


# weight decay for preidction layer
if hasattr(config.model, 'lr_pred'):
    args.lr_pred = config.model.lr_pred

# setting learning rate values
if hasattr(config.model, 'lr_attn') and hasattr(config.model, 'lr_fuse') and hasattr(config.model, 'lr_loss'):
    args.lr_attn = config.model.lr_attn
    args.lr_fuse = config.model.lr_fuse
    args.lr_loss = config.model.lr_loss
    pass
else:
    if hasattr(config.model, 'lr'):
        args.lr_attn = args.lr_fuse = args.lr_pred = args.lr_loss = config.model.lr

# learning rate for preidction layer
if hasattr(config.model, 'lr_pred'):
    args.lr_pred = config.model.lr_pred

if args.resume_train:
    args.warmup_epochs = 0

if args.region_size is not None:
    args.region_length = args.region_size // args.patch_size
else:
    args.region_length = 0

# Checking to see if region-size, region-length, and grid-size are valid
# These parameters control the subdivision of patches within a given region
if args.region_length is not None and args.region_length > 0:
    assert_message = "grid size is measured in patches and need to be a positive number no larger than the region size / patch size"
    #TODO: Use multi-line comment.
    assert args.grid_size <= args.region_length and args.grid_size > 0, assert_message

args.prop_mask = [int(x) for x in args.prop_mask.split(',')]
args.prop_mask = [x / sum(args.prop_mask) for x in args.prop_mask]

# initializing sampling and outcome arguments
#TODO: lack of docs. So I guess this is a variable name in string for id column?
if args.sample_svs:
    args.id_var = 'id_svs_num'
else:
    args.id_var = 'id_patient_num'

if config.dataset.outcome_type == 'survival':
    args.outcomes = ['time', 'status']
else:
    args.outcomes = [config.dataset.outcome]

args.patch_spec = f"mag_{float(config.patch.magnification):.1f}-size_{args.patch_size}"

#TODO: This should be in a function.
args.mode_ops = {'train': {}, 'val': {}, 'predict': {}}

# initializing num_patches argument for train mode
if config.patch.num_patches > 0:
    args.mode_ops['train']['num_patches'] = config.patch.num_patches
else:
    if args.region_length is None:
        args.mode_ops['train']['num_patches'] = 0
    else:
        args.mode_ops['train'][
            'num_patches'] = args.region_length * args.region_length

# initializing num_patches argument for validation mode
if args.num_patches_val is None:
    args.mode_ops['val']['num_patches'] = args.mode_ops['train']['num_patches']
elif args.num_patches_val > 0:
    args.mode_ops['val']['num_patches'] = args.num_patches_val
else:
    args.mode_ops['val'][
        'num_patches'] = args.region_length * args.region_length

args.mode_ops['predict']['num_patches'] = args.mode_ops['val']['num_patches']

# number of regions to sample from in train mode
args.mode_ops['train']['num_regions'] = config.model.regions_per_svs

# number of regions to sample from in validation mode
if args.regions_per_svs_val is None:
    args.mode_ops['val']['num_regions'] = config.model.regions_per_svs
else:
    args.mode_ops['val']['num_regions'] = args.regions_per_svs_val

# number of regions to sample from in predict mode
args.mode_ops['predict']['num_regions'] = args.mode_ops['val']['num_regions']

# number of svs to sample from in train, val, and predict modes
args.mode_ops['train']['svs_per_patient'] = args.svs_per_patient
args.mode_ops['val']['svs_per_patient'] = args.svs_per_patient
args.mode_ops['predict']['svs_per_patient'] = args.svs_per_patient

# regions_per_patient dervied from the above parameters
args.mode_ops['train'][
    'regions_per_patient'] = config.model.regions_per_svs * args.svs_per_patient
args.mode_ops['val']['regions_per_patient'] = args.mode_ops['val'][
    'num_regions'] * args.svs_per_patient
args.mode_ops['predict']['regions_per_patient'] = args.mode_ops['val'][
    'regions_per_patient']

# setting remaining arguments based on modes
args.mode_ops['train']['repeats_per_epoch'] = args.repeats_per_epoch
args.mode_ops['val']['repeats_per_epoch'] = 1
args.mode_ops['predict']['repeats_per_epoch'] = args.repeats_per_epoch

args.mode_ops['train']['batch_size'] = max(config.model.batch_size,
                                           args.svs_per_patient)
args.mode_ops['val']['batch_size'] = max(config.model.batch_size, args.svs_per_patient)
args.mode_ops['predict']['batch_size'] = max(config.model.batch_size,
                                             args.svs_per_patient)

if args.visualization:
    args.vis_spec = f"{args.timestr}-{config.model.resume}/{args.vis_layer}-{args.vis_head}"

# sets the current working directory to the directory where the script is located
script_path = Path(__file__).resolve()
script_dir = script_path.parent
os.chdir(script_dir)

def get_checkpoint_epoch(fname):
    """
    Args:
        fname (str): checkpoint filename e.g. checkpoints/pretrained_20x_448_resnet34/0500.pt
    Returns:
        int: epoch number
    """
    return os.path.basename(fname).split(".")[0]


def get_resume_checkpoint(checkpoints_name, epoch_to_resume):
    """
    Args:
        checkpoints_name (str): name of the checkpoints folder e.g. pretrained_20x_448_resnet34
    Returns:
        str: checkpoint filename e.g. checkpoints/pretrained_20x_448_resnet34/0500.pt
    """
    files = glob.glob(
        os.path.join(args.checkpoints_folder, checkpoints_name, "*.pt"))

    #TODO: Avoid list comprehension
    checkpoint_to_resume = [
        fname for fname in files
        if get_checkpoint_epoch(fname) == epoch_to_resume
    ][0]
    
    return checkpoint_to_resume


def prepare_data(meta_split, meta_file, vars_to_include=[]):
    """
    Merge and preprocess meta_split and meta_file dataframes for use by the model.

    Parameters
    ----------
    meta_split : pandas.DataFrame
        A pandas dataframe containing the split information for each patient. The dataframe must contain information about patient ids

    meta_file : pandas.DataFrame
        A pandas dataframe containing the patient metadata. The dataframe must contain a column named 'id_patient'

    vars_to_include : list, optional
        A list of additional variables to include in the merged dataframe. Default is an empty list.

    Returns
    -------
    pandas.DataFrame
        A merged and preprocessed pandas dataframe containing the split information and patient metadata. The returned dataframe may contain the following columns:
        - id_patient: unique patient ID
        - case_number: patient case number (may need to be corrected - showing as 'SP' for IBD dataset)
        - id_patient_num: encoded patient ID
        - id_svs_num: encoded id_svs
        - outcome: patient outcome variable, encoded for classification models e.g. 0, 1, 2 for three classes

    """

    #TODO: Duct-tape solution for RCCp dataset. Need update.
    ids_to_add = []
    for index, row in meta_split.iterrows():
        if 'case_number' in row:
            value_to_split = row['case_number']
            split_value = value_to_split.split('.')[0]
        elif 'barcode' in row:
            split_value = row['barcode']
            print(row)
        else:
            raise ValueError("Row does not contain 'case_number' or 'barcode'")

        ids_to_add.append(split_value)
    meta_split['id_patient'] = ids_to_add
    # Define the lambda function
    shorten_id = lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2]
    # Apply the lambda function to the id_patient column
    meta_file['id_patient'] = meta_file['id_patient'].apply(shorten_id)



    #TODO: This whole block should be in another script to be run before train.py
    if 'id_patient' not in meta_split.columns:
        patient_ids = []
        # iterating over the meta_split dataframe
        for index, row in meta_split.iterrows():
            #TODO: Debug: This function is basically repeating what we have done in MaskHIT_Prep
            print(row)
            # obtaining the paths of the files to the related slide
            
            #Debug: temp fix
            #file_names = ast.literal_eval(row['Path'])
            file_names = str(row['path'])

            patient_id = file_names[0].split('/')[5].split(' ')[0]
            patient_ids.append(patient_id) # adding patient id to the list
        meta_split['id_patient'] = patient_ids # adding column to the meta_split dataframe
        # formatting rows in meta_file of the id patients so they match that of meta_split df
        meta_file['id_patient'] = meta_file['id_patient'].apply(lambda x: pd.Series(x.split(' ')[0]))


    vars_to_keep = ['id_patient']
    if config.dataset.outcome_type in ['survival', 'mlm']:
        vars_to_keep.extend(['time', 'status'])
    else:
        vars_to_keep.append(config.dataset.outcome)

    # Selects columns from meta_file df and merges them into meta_split based on a shared 'id_patient' column
    # includes all the columns from meta_split and only the selected columns from meta_file
    meta_split = meta_split.merge(meta_file[vars_to_include],
                                  on='id_patient',
                                  how='inner')
    #TODO: Need check consistency after merge.
    assert meta_split.shape[0] > 0, "Merge operation"

    
    meta_split['id_patient_num'] = meta_split.id_patient.astype(
        'category').cat.codes
    meta_split['id_svs_num'] = meta_split.id_svs.astype('category').cat.codes

    # converting the outcome variable to numerical value
    if config.dataset.outcome_type == 'classification':
        meta_split = meta_split.loc[~meta_split[config.dataset.outcome].isna()]
        meta_split[config.dataset.outcome] = meta_split[config.dataset.outcome].astype(
            'category').cat.codes
    elif config.dataset.outcome_type == 'survival':
        meta_split = meta_split.loc[~meta_split.status.isna()
                                    & ~meta_split.time.isna()]
    elif config.dataset.outcome_type == 'regression':
        meta_split = meta_split.loc[~meta_split[config.dataset.outcome].isna()]
        meta_split[config.dataset.outcome] = meta_split[config.dataset.outcome].astype(
            'float')
    
    return meta_split


def main():

    #TODO: Debug
    print()

    if len(args.timestr):
        TIMESTR = args.timestr
    else:
        TIMESTR = time.strftime("%Y%m%d_%H%M%S")
    model_name = str(TIMESTR)

    if config.dataset.meta_all is not None:
        model_name = f"{TIMESTR}-{args.fold}"

    # if we want to resume previous training
    if config.model.resume:
        checkpoint_to_resume = get_resume_checkpoint(config.model.resume,
                                                     config.model.resume_epoch)
        if args.resume_train:
            # use the model name
            model_name = config.model.resume
            TIMESTR = model_name.split('-')[0]

    # if we want to resume when error occurs
    elif args.resume_train:
        args.resume = model_name
        try:
            checkpoint_to_resume = get_resume_checkpoint(args.resume, "LAST")
        except Exception as e:
            print(e)
            checkpoint_to_resume = ''

    # we don't want to resume
    else:
        checkpoint_to_resume = ''

    args.model_name = model_name

    # setting the checkpoints folder with the name of model (including date and fold)
    checkpoints_folder = os.path.join(args.checkpoints_folder, model_name)
    args.hostname = socket.gethostname()

    # loading datasets
    meta_svs = pd.read_pickle(config.dataset.meta_svs) 

    #TODO: Need doc why this option exists and what it does.
    if args.ffpe_only:
        meta_svs = meta_svs.loc[meta_svs.slide_type == 'ffpe']
    if args.ffpe_exclude:
        meta_svs = meta_svs.loc[meta_svs.slide_type != 'ffpe']

    if config.dataset.meta_all is not None:
        meta_all = pd.read_pickle(config.dataset.meta_all)
        #Debug:
        print("Debug: meta_all:\n", meta_all.head(2))

        #TODO: mode=extract is not expected in the train_options. Need doc.
        if args.mode == 'extract':
            meta_train = meta_val = meta_all
        elif 'fold' in meta_all.columns:
            if meta_all.fold.nunique() == 5:
                #TODO: Why expecting to have 5 folds? Generalize.
                val_fold = (args.fold + 1) % 5
                test_fold = args.fold

                # train_folds are the folds not used for validation or testing
                train_folds = [
                    x for x in range(5) if x not in [val_fold, test_fold]
                ]

            # corresponds to rows that belong to folds in train_folds
            meta_train = meta_val = meta_all.loc[meta_all.fold.isin(
                train_folds)]
            if args.test_type == 'train':
                meta_val = meta_train
            elif args.test_type == 'val':
                meta_val = meta_all.loc[meta_all.fold == val_fold]
            elif args.test_type == 'test':
                meta_val = meta_all.loc[meta_all.fold == test_fold]

        else:
            meta_train = meta_all.loc[meta_all.train]
            if args.test_type == 'train':
                meta_val = meta_all.loc[meta_all.train]
            elif args.test_type == 'val':
                meta_val = meta_all.loc[meta_all.val]
            elif args.test_type == 'test':
                meta_val = meta_all.loc[meta_all.test]

    else:
        meta_train = pd.read_pickle(args.meta_train)
        meta_val = pd.read_pickle(args.meta_val)

    # select cancer subset
    #TODO: lack of docs. cancer='' means use all. I think right term would be 'filtering'.
    if args.cancer == '':
        pass
    else:
        meta_svs = meta_svs.loc[meta_svs.cancer == args.cancer]
        meta_train = meta_train.loc[meta_train.cancer == args.cancer]
        meta_val = meta_val.loc[meta_val.cancer == args.cancer]


    #TODO: Not sure why we need this.
    if config.dataset.is_cancer:
        meta_svs['folder'] = meta_svs['cancer']
    else:
        meta_svs['folder'] = config.dataset.disease
    
    #TODO: Is safe to hard-code the weights?
    meta_svs['sampling_weights'] = 1
    vars_to_include = ['id_patient', 'folder', 'id_svs', 'sampling_weights']
    if 'svs_path' in meta_svs:
        vars_to_include = ['id_patient', 'folder', 'id_svs', 'sampling_weights', 'svs_path']

    if args.visualization and 'pos' in meta_svs.columns:
        vars_to_include.append('pos')

    ########################################
    # prepare dataset
    df_test = prepare_data(meta_split=meta_val, 
                           meta_file=meta_svs,  
                           vars_to_include=vars_to_include) 
    df_train = prepare_data(meta_split=meta_train,  
                            meta_file=meta_svs, 
                            vars_to_include=vars_to_include)

    print("df_test")
    print(df_test)
    
    if config.dataset.outcome_type == 'classification':
        num_classes = len(df_train[config.dataset.outcome].unique().tolist())
    else:
        num_classes = 1
    print(f"num_classes: {num_classes}")
    
    if config.model.weighted_loss:
        weight = df_train.shape[0] / df_train[
            config.dataset.outcome].value_counts().sort_index()
    else:
        weight = None
    
    criterion = FlexLoss(outcome_type=config.dataset.outcome_type, weight=weight)

    if config.dataset.study is not None:
        model_name = f"{config.dataset.study}/{model_name}"

    # initializing a fitter by passing in above arguments and loss functions
    hf = HybridFitter(timestr=TIMESTR,
                      num_classes=num_classes,
                      args=args,
                      config_file = config,
                      loss_function=criterion,
                      model_name=model_name,
                      checkpoints_folder=checkpoints_folder,
                      checkpoint_to_resume=checkpoint_to_resume)


    data_dict = {"train": df_train, "val": df_test}

    df_test.to_csv('fold0.csv')

    # Simply call main_worker function
    # print(f"Validation Folds: {df_test.fold.unique()}")
    if args.mode == 'test':
        hf.fit(data_dict, 'test')

    elif args.mode == 'train':
        hf.fit(data_dict)

    elif args.mode == 'predict':
        hf.predict(df_test)

    elif args.mode == 'extract':
        hf.fit(data_dict, procedure='extract')

    else:
        print(f"Mode {args.mode} has not been implemented!")

if __name__ == '__main__':
    main()
    
