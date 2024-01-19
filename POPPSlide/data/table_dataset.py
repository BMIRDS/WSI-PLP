from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from data.dataset_helper import random_crops

########################################
# load slide patches
class SlideDatasetFromTable(Dataset):
    def __init__(
            self,
            data_file,  #FIXME: RENAME TO DATA_FRAME
            image_dir,
            outcome,
            outcome_type,
            crop_size,
            num_crops,
            features=None,
            transform=None,
            ):

        self.df = data_file
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.transform = transform

        if outcome_type == 'survival':
            self.outcomes = self.df[['time', 'status']].to_numpy().astype(float)
        else:
            #CLASSIFICATION OR REGRESSION
            self.outcomes = self.df[[outcome]].to_numpy().astype(float)
        self.ids = self.df['id_patient'].tolist()
        self.files = self.df['file'].tolist()

        try:
            self.random_seeds = self.df['random_id'].tolist()
        except:
            self.random_seeds = np.ones(self.df.shape[0])

    def __len__(self):
        return self.df.shape[0]

    def sample_patch(self, idx):
        idx = idx % self.df.shape[0]
        fname = self.files[idx]
        random_seed = self.random_seeds[idx]

        img_name = Path(self.image_dir) / fname
        imgs = np.array(Image.open(img_name))
        imgs = random_crops(imgs, self.crop_size, self.num_crops).reshape(-1, self.crop_size, 3)
        if self.transform is None:
            pass
        else:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            imgs = self.transform(torch.tensor(imgs).permute(2,0,1)/255.)
        sample = (
            imgs, 
            self.ids[idx], 
            # random_seed, 
            self.outcomes[idx,:]
            )
        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.sample_patch(idx)
