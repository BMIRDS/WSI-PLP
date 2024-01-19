from pathlib import Path
from PIL import Image
import numpy as np
import os
import pickle
import random

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torchvision.transforms.functional as tv_F

from data.list_dataset import SlideDatasetFromList, PatientDatasetFromList
from data.table_dataset import SlideDatasetFromTable

"""NOTE: MAINLY STORE DATALOADERS
BY INSTANTIATE CORRESPONDING DATASETS
"""

def prepare_table_datasets(
        args,
        pickle_file,
        transform,
        mode='train',):
    if isinstance(pickle_file, str):
        _df = pd.read_pickle(pickle_file)
    else:
        _df = pickle_file

    meta_df = {}

    if mode == 'train':
        if args.sampling_ratio is None:
            sampling_ratio = None
        else:
            sampling_ratio = list(map(int, args.sampling_ratio.split(',')))
        meta_df[mode] = grouped_sample(
                _df, 
                stratify_var=args.stratify, 
                weights=sampling_ratio, 
                num_obs=len(_df.submitter_id.unique())*args.repeats_per_epoch,
                num_patches=args.num_patches,
                patient_var='submitter_id')

        # writer['meta'].info(meta_df[mode].shape)
    elif args.sample_id:
        meta_df[mode] = _df.groupby('submitter_id', group_keys=False).apply(
            lambda x: x.sample(args.num_val, replace=True))
    else:
        meta_df[mode] = _df


    if mode == 'train':
        batch_size = args.num_patches * args.batch_size
        num_crops = args.num_crops
    elif mode == 'val':
        batch_size = args.num_val
        num_crops = args.num_crops
    else:
        batch_size = args.batch_size
        num_crops = args.num_crops

    data = SlideDatasetFromTable(
        data_file=meta_df[mode],
        image_dir='./',
        crop_size=args.crop_size,
        num_crops=num_crops,
        outcome=args.outcome,
        outcome_type=args.outcome_type,
        transform=transform[mode]
    )

    return data

    


def get_dataloader(args, transform):
    '''
    Constructed dataloader for training and validation data, currently called in main
    config: arguments passed from configuration file
    transform: data augmentation from get_transformation function
    '''
    if args.config is None:
        ########################################
        # prepare dataset
        df_patches = pd.read_pickle(args.patch_meta)
        df_patches['id_patient'] = df_patches.submitter_id.astype('category').cat.codes
        df_train   = pd.read_pickle(args.train_meta)
        df_test    = pd.read_pickle(args.test_meta )

        cols_to_use = df_patches.columns.difference(df_train.columns)
        cols_to_use = cols_to_use.tolist()
        cols_to_use.append('submitter_id')
        df_train = df_train.merge(df_patches[cols_to_use], on='submitter_id', how='left')
        df_test  = df_test.merge( df_patches[cols_to_use], on='submitter_id', how='left')

        if args.outcome_type == 'survival':
            outcomes = ['time','status']
        else:
            outcomes = [args.outcome]

        df_train.dropna(subset=outcomes, inplace=True)
        df_test.dropna( subset=outcomes, inplace=True)


        # prepare_datasets
        train_data = prepare_table_datasets(args, df_train, transform, 'train')
        val_data = prepare_table_datasets(args, df_test, transform, 'val')

        train_loader = DataLoader(
            train_data,
            shuffle=False,
            batch_size=args.batch_size * args.num_patches,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True
        )

        val_loader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=False
        )

        loader = {'train':train_loader, 'val':val_loader}
        return loader

    else:
        ### Read data split
        with open(args.data_split, 'rb') as handle:
            data_split = pickle.load(handle)
        
        if args.train_level == 'patient':
            drop_last = False
            train_data = PatientDatasetFromList(
                patient_paths=data_split['train'], num_patches = args.num_patches,
                transform = transform['train'], class_label = args.class_label)
            val_data = PatientDatasetFromList(
                patient_paths=data_split['val'], num_patches = args.num_val, 
                transform = transform['val'], class_label = args.class_label)
            
        elif args.train_level == 'slide':
            drop_last = True
            
            train_data = SlideDatasetFromList(
                patient_paths=data_split['train'], num_patches = args.num_patches,
                transform = transform['train'], class_label = args.class_label)
            val_data = SlideDatasetFromList(
                patient_paths=data_split['val'], num_patches = args.num_val,
                transform = transform['val'], class_label = args.class_label)

        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4, 
            drop_last=drop_last)
        val_loader = DataLoader(
            val_data,
            batch_size = 1,
            shuffle=True,
            num_workers=args.num_workers)
        loader = {'train':train_loader, 'val':val_loader}
        return loader


def weighted_sampling(df, num_obs, weights = [3,1], stratify_var='status', patient_var='submitter_id'):
    '''
    Weighted sampling to sample both positive and negative class from batch
    Only used for Survival data
    TODO: revisit for suvival analysis
    '''
    ids = df.groupby(stratify_var)[patient_var].apply(unique_shuffle_to_list).tolist()
    groups = []
    while len(groups) < num_obs:
        group = []
        for j,k in enumerate(weights):
            while k > 0:
                if min(list(map(len,ids))) == 0:
                    ids = df.groupby(stratify_var)[patient_var].apply(unique_shuffle_to_list).tolist()
                group.append(ids[j].pop())
                k -= 1
        np.random.shuffle(group)
        groups.extend(group)
        
    groups = groups[:num_obs]
    dfg = pd.DataFrame(groups,columns=[patient_var])
    dfg['queue_order'] = dfg.index
    dfg = dfg.merge(df,on=patient_var).sort_values('queue_order').reset_index(drop=True)
    return dfg

def unique_shuffle_to_list(x):
    x = x.unique()
    np.random.shuffle(x)
    return x.tolist()

def grouped_sample(data, stratify_var, weights, num_obs=10000, num_patches=4, patient_var='submitter_id', patch_var='file'):
    #################################
    # step 1: sample patients
    if stratify_var is None:
        data_meta = data[[patient_var]].drop_duplicates()
        groups = []
        while len(groups) < num_obs:
            groups.extend(data_meta.sample(frac=1.)[patient_var].tolist())
    else:
        data_meta = data[[patient_var,stratify_var]].drop_duplicates()
        ids = data_meta.groupby(stratify_var)[patient_var].apply(unique_shuffle_to_list).tolist()
        groups = []
        while len(groups) < num_obs:
            group = []
            for j,k in enumerate(weights):
                while k > 0:
                    if min(list(map(len,ids))) == 0:
                        ids = data_meta.groupby(stratify_var)[patient_var].apply(unique_shuffle_to_list).tolist()
                    group.append(ids[j].pop())
                    k -= 1
            np.random.shuffle(group)
            groups.extend(group)
    # post processing
    groups = groups[:num_obs]

    dfg = pd.DataFrame(groups,columns=[patient_var])
    dfg['queue_order'] = dfg.index
    dfg = dfg.merge(data_meta,on=patient_var).sort_values('queue_order').reset_index(drop=True)

    #################################
    # step 2: sample patches
    # get the order of occurrence for each patient
    dfg['dummy_count'] = 1
    dfg['within_index'] = dfg.groupby(patient_var).dummy_count.cumsum() - 1

    # sample sufficient amount of patches
    dfp = data.groupby(patient_var, as_index=False).sample(
        num_patches * (dfg.within_index.max() + 1), replace=True)
    # for each patient, determine merge to which occurrence
    dfp['within_index'] = dfp.groupby(patient_var)[patch_var].transform(
        lambda x: np.arange(x.shape[0]) // num_patches)

    if stratify_var is not None:
        dfg.drop([stratify_var],axis=1,inplace=True)
    df_sel = dfg.merge(dfp, on=[patient_var, 'within_index'], how='left')
    return df_sel


