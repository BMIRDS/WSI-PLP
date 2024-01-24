import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from maskhit.trainer.wsitilesampler import WsiTileSampler
from maskhit.trainer.helper import zero_padding

###########################################
#          CUSTOM DATALOADER              #
###########################################


def restore_mask(data):
    mask = np.zeros((data.pos_x.max() + 1, data.pos_y.max() + 1))
    for i, row in data.loc[data.valid == 1].iterrows():
        mask[row['pos_x'], row['pos_y']] = 1
    return mask


def load_features(features, fids):
    fids = fids.long()
    dim = features.size(1)
    return torch.cat([features, torch.zeros(1, dim).to(features.device)])[fids]


def get_image_fname(id_svs, loc, magnification=20, patch_size=224):
    return f"patches/mag_{magnification}-size_{patch_size}/{id_svs}/{loc[0]:05d}/{loc[1]:05d}.jpeg"


def load_images(id_svs, locs, is_valid, magnification=20, patch_size=224):
    imgs = []
    for loc, valid in zip(locs, is_valid):
        if valid:
            img_fname = get_image_fname(id_svs=id_svs,
                                        loc=loc,
                                        magnification=magnification,
                                        patch_size=patch_size)
            imgs.append(np.array(Image.open(img_fname)))
        else:
            imgs.append(np.zeros((3, patch_size, patch_size)))
    return imgs


class SlidesDataset(Dataset):
    '''
    sample patches or features from the datasets
    '''

    def __init__(self,
                 data_file,
                 outcomes,
                 writer=None,
                 mode='train',
                 transforms=None,
                 num_regions=1,
                 num_patches=100,
                 region_length=None,
                 args=None):

        self.df = data_file

        self.args = args
        self.writer = writer

        if self.args.outcome_type == 'mlm':
            self.outcomes = np.ones((self.df.shape[0], 1))
        else:
            self.outcomes = self.df[outcomes].to_numpy().astype(float)

        self.ids = self.df[self.args.id_var].tolist()
        self.files = self.df['id_svs'].tolist()
        self.folders = self.df['folder'].tolist()
        if args.visualization:
            print(self.df.keys())
            self.locs = self.df['pos'].tolist()
        else:
            self.locs = [None for _ in range(self.df.shape[0])]

        self.wsi = None
        self.sample_size = self.df.shape[0]
        self.mode = mode
        self.transforms = transforms
        self.num_regions = num_regions
        self.region_length = region_length
        self.num_patches = num_patches
        self.sample_all = num_patches == (region_length * region_length)

    def __len__(self):
        return self.sample_size

    def _get_patch_meta(self, folder, fname, loc):
        # get all the patches for one wsi
        #TODO: This path doesn't seem to be well generalized.
        #Confusing as folder=args.config.disease.
        meta_one = pd.read_pickle(
            f'{self.args.data}/{folder}/{fname}/{self.args.patch_spec}/meta.pickle')
        meta_one['valid'].fillna(0, inplace=True)
        meta_one.reset_index(drop=True, inplace=True)

        # get all the valid features for one wsi
        wsi = WsiTileSampler(meta_one,
                             sample_all=self.sample_all,
                             mode=self.mode,
                             num_patches=self.num_patches,
                             args=self.args)

        output = wsi.sample(self.num_patches,
                            self.region_length,
                            num_regions=self.num_regions,
                            threshold=self.args.sampling_threshold,
                            weighted_sample=False,
                            loc=loc)
        output['is_valid'] = output['fids'] > -1
        # print("Printing output['is_valid'] in _get_patch_meta of SlideDataset class")
        # print(output['fids'])
        return output

    def sample_features(self, folder, svs_identifier, loc):
        features_one = torch.load(
            f'{self.args.data}/{folder}/{svs_identifier}/{self.args.patch_spec}/{self.args.backbone}/features.pt'
        ).detach()
        tile_one = self._get_patch_meta(folder=folder,
                                        fname=svs_identifier,
                                        loc=loc)
        tile_one['imgs'] = load_features(features_one, tile_one['fids'])
        return tile_one

    def sample_patches(self, folder, svs_identifier):
        tile_one = self._get_patch_meta(folder=folder, fname=svs_identifier)
        is_valid = tile_one['is_valid']
        imgs = load_images(svs_identifier, tile_one['locs_orig'], is_valid)
        processed = []
        if self.transforms is not None:
            for img, valid in zip(imgs, is_valid):
                if valid:
                    processed.append(self.transforms(img))
                else:
                    processed.append(img)
        processed = torch.stack(processed)
        tile_one['imgs'] = processed
        return tile_one

    def sample_patch(self, idx):
        tiles = torch.zeros(0)
        idx = idx % self.df.shape[0]
        svs_identifier = self.files[idx]
        folder = self.folders[idx]
        loc = self.locs[idx]

        if self.args.use_patches:
            one_sample = self.sample_patches(folder, svs_identifier, loc)
        else:
            one_sample = self.sample_features(folder, svs_identifier, loc)

        id = self.ids[idx]
        outcome = self.outcomes[idx, :]
        imgs, pos_tile, pos, pct_valid = one_sample['imgs'], one_sample[
            'loc_tiles'], one_sample['locs_local'], one_sample['pct_valid']

        if self.args.visualization:
            sample = (imgs,
                      id,
                      outcome,
                      pos,
                      pos_tile,
                      tiles,
                      pct_valid)
        else:
            sample = (zero_padding(imgs, self.num_regions, 0), id, outcome,
                      zero_padding(pos, self.num_regions, 0),
                      zero_padding(pos_tile, self.num_regions, 0),
                      zero_padding(tiles, self.num_regions, 0),
                      zero_padding(pct_valid, self.num_regions, 0))
        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.sample_patch(idx)
