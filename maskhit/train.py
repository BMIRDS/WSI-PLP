"""
USAGE: python train.py --user-config-file <filename> --sample-patient --batch-size <value> --override-logs --timestr <value> --fold <value>

REQUIRED OPTIONS:
  --user-config-file <filename>    Specify the user configuration file
  --fold <value>                   Specify the fold number

EXAMPLES:
- For IBD project
python train.py --user-config-file configs/config_ibd_train.yml --sample-patient -b=4 --override-logs --timestr=2023_5_30 --fold=0
"""

import os
import time
import ast
import pandas as pd
import sys
import glob
import socket
from maskhit.trainer.fitter import HybridFitter
from maskhit.trainer.losses import FlexLoss
from options.train_options import TrainOptions
from utils.config import Config

print("\n")
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

args = opt.parse()
print(f"args: {args}")

# args_config = default_options()
config = Config(args.user_config_file)

print(f"TEST: {config.dataset.meta_svs}")

args.all_arguments = ' '.join(sys.argv[1:])


assert not args.sample_all, "the argument --sample-all is deprecated, use --num-patches=0 instead"

print(f"args.cancer: {args.cancer}")
if args.cancer == '.':
    args.cancer = ""

if config.patch.wd is not None:
    args.wd_attn = args.wd_fuse = args.wd_pred = args.wd_loss = config.patch.wd

if args.lr is not None:
    config.model.lr_attn = args.lr_fuse = config.model.lr_pred = args.lr_loss = args.lr



if args.resume_train:
    args.warmup_epochs = 0

if args.region_size is not None:
    args.region_length = args.region_size // args.patch_size
else:
    args.region_length = 0

if args.region_length is not None and args.region_length > 0:
    assert_message = "grid size is measured in patches and need to be a positive number no larger than the region size / patch size"
    assert args.grid_size <= args.region_length and args.grid_size > 0, assert_message

args.prop_mask = [int(x) for x in args.prop_mask.split(',')]
args.prop_mask = [x / sum(args.prop_mask) for x in args.prop_mask]

if args.sample_svs:
    args.id_var = 'id_svs_num'
else:
    args.id_var = 'id_patient_num'

if config.dataset.outcome_type == 'survival':
    args.outcomes = ['time', 'status']
else:
    args.outcomes = [config.dataset.outcome]


# args.patch_spec = f"mag_{str(args.magnification) + '.0'}-size_{args.patch_size}"
args.patch_spec = f"mag_{float(config.patch.magnification):.1f}-size_{args.patch_size}"


args.mode_ops = {'train': {}, 'val': {}, 'predict': {}}

if config.patch.num_patches > 0:
    args.mode_ops['train']['num_patches'] = config.patch.num_patches
else:
    if args.region_length is None:
        args.mode_ops['train']['num_patches'] = 0
    else:
        args.mode_ops['train'][
            'num_patches'] = args.region_length * args.region_length

if args.num_patches_val is None:
    args.mode_ops['val']['num_patches'] = args.mode_ops['train']['num_patches']
elif args.num_patches_val > 0:
    args.mode_ops['val']['num_patches'] = args.num_patches_val
else:
    args.mode_ops['val'][
        'num_patches'] = args.region_length * args.region_length

args.mode_ops['predict']['num_patches'] = args.mode_ops['val']['num_patches']

args.mode_ops['train']['num_regions'] = args.regions_per_svs
if args.regions_per_svs_val is None:
    args.mode_ops['val']['num_regions'] = args.regions_per_svs
else:
    args.mode_ops['val']['num_regions'] = args.regions_per_svs_val
args.mode_ops['predict']['num_regions'] = args.mode_ops['val']['num_regions']

args.mode_ops['train']['svs_per_patient'] = args.svs_per_patient
args.mode_ops['val']['svs_per_patient'] = args.svs_per_patient
args.mode_ops['predict']['svs_per_patient'] = args.svs_per_patient

args.mode_ops['train'][
    'regions_per_patient'] = args.regions_per_svs * args.svs_per_patient
args.mode_ops['val']['regions_per_patient'] = args.mode_ops['val'][
    'num_regions'] * args.svs_per_patient
args.mode_ops['predict']['regions_per_patient'] = args.mode_ops['val'][
    'regions_per_patient']

args.mode_ops['train']['repeats_per_epoch'] = args.repeats_per_epoch
args.mode_ops['val']['repeats_per_epoch'] = 1
args.mode_ops['predict']['repeats_per_epoch'] = args.repeats_per_epoch

args.mode_ops['train']['batch_size'] = max(args.batch_size,
                                           args.svs_per_patient)
args.mode_ops['val']['batch_size'] = max(args.batch_size, args.svs_per_patient)
args.mode_ops['predict']['batch_size'] = max(args.batch_size,
                                             args.svs_per_patient)

if args.visualization:
    args.vis_spec = f"{args.timestr}-{args.resume}/{args.vis_layer}-{args.vis_head}"

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def get_checkpoint_epoch(fname):
    return os.path.basename(fname).split(".")[0]


def get_resume_checkpoint(checkpoints_name, epoch_to_resume):
    files = glob.glob(
        os.path.join(args.checkpoints_folder, checkpoints_name, "*.pt"))
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

    if 'id_patient' not in meta_split.columns:
        patient_ids = []
        # iterating over the meta_split dataframe
        for index, row in meta_split.iterrows():
            # obtaining the paths of the files to the related slide
            file_names = ast.literal_eval(row['Path'])
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
    
    print("meta_split = ", meta_split.columns, meta_split.shape)
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
    
    return meta_split


def main():

    if len(args.timestr):
        TIMESTR = args.timestr
    else:
        TIMESTR = time.strftime("%Y%m%d_%H%M%S")
    model_name = str(TIMESTR)
    if config.dataset.meta_all is not None:
        model_name = f"{TIMESTR}-{args.fold}"

    # if we want to resume previous training
    if len(args.resume):
        checkpoint_to_resume = get_resume_checkpoint(args.resume,
                                                     args.resume_epoch)
        if args.resume_train:
            # use the model name
            model_name = args.resume
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

    checkpoints_folder = os.path.join("checkpoints", model_name)
    args.hostname = socket.gethostname()

    # loading datasets
    meta_svs = pd.read_pickle(config.dataset.meta_svs) 

    if args.ffpe_only:
        meta_svs = meta_svs.loc[meta_svs.slide_type == 'ffpe']
    if args.ffpe_exclude:
        meta_svs = meta_svs.loc[meta_svs.slide_type != 'ffpe']

    if config.dataset.meta_all is not None:
        meta_all = pd.read_pickle(config.dataset.meta_all)
        if args.mode == 'extract':
            meta_train = meta_val = meta_all
        elif 'fold' in meta_all.columns:
            if meta_all.fold.nunique() == 5:
                val_fold = (args.fold + 1) % 5
                test_fold = args.fold
                train_folds = [
                    x for x in range(5) if x not in [val_fold, test_fold]
                ]


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
    if args.cancer == '':
        pass
    else:
        meta_svs = meta_svs.loc[meta_svs.cancer == args.cancer]
        meta_train = meta_train.loc[meta_train.cancer == args.cancer]
        meta_val = meta_val.loc[meta_val.cancer == args.cancer]

    print('shape of meta_svs = ', meta_svs.shape)

    if config.dataset.is_cancer:
        meta_svs['folder'] = meta_svs['cancer']
    else:
        meta_svs['folder'] = config.dataset.disease
    
    meta_svs['sampling_weights'] = 1
    vars_to_include = ['id_patient', 'folder', 'id_svs', 'sampling_weights']
    if 'svs_path' in meta_svs:
        vars_to_include = ['id_patient', 'folder', 'id_svs', 'sampling_weights', 'svs_path']

    print('=' * 30)
    print(meta_svs.columns)
    print('=' * 30)
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

    
    print("TRAINING DATA")
    print(df_train)
    print("TESTING DATA")
    print(df_test)

    if config.dataset.outcome_type == 'classification':
        num_classes = len(df_train[config.dataset.outcome].unique().tolist())
    else:
        num_classes = 1
    
    print("NUM CLASSES: " + str(num_classes))

    print('num_classes = ', num_classes)
    if args.weighted_loss:
        weight = df_train.shape[0] / df_train[
            config.dataset.outcome].value_counts().sort_index()
        print('weight is: ', weight)
    else:
        weight = None
    criterion = FlexLoss(outcome_type=config.dataset.outcome_type, weight=weight)

    if config.dataset.study is not None:
        model_name = f"{config.dataset.study}/{model_name}"

    hf = HybridFitter(timestr=TIMESTR,
                      num_classes=num_classes,
                      args=args,
                      config_file = config,
                      loss_function=criterion,
                      model_name=model_name,
                      checkpoints_folder=checkpoints_folder,
                      checkpoint_to_resume=checkpoint_to_resume)


    data_dict = {"train": df_train, "val": df_test}

    # Simply call main_worker function
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