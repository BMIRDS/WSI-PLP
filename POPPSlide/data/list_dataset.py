from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch

from torch.utils.data import Dataset

##########################################
####        Dataset Class             ####
##########################################

class PatientDatasetFromList(Dataset):
    # TODO: Comment and check functions
    def __init__(self, patient_paths, num_patches, transform=None, class_label=None):
        self.random_seed = random.sample(range(0, 10000), len(patient_paths))
        self.patient_paths = patient_paths
        self.num_patches = num_patches
        self.transform = transform
        self.class_label = class_label # {'nodepos':1, 'nodeneg':0}
        
    def __getitem__(self, idx):
        patient_path = self.patient_paths[idx]
        label = self.class_label[patient_path.parents[0].name]
        images = self.read_random_patches(list(patient_path.glob('*/*.png')))        
        if self.transform:
            images_transform = []
            for patch in images:
                random.seed(self.random_seed[idx])
                torch.manual_seed(self.random_seed[idx])
                images_transform.append(self.transform(patch))
            
            images = torch.stack([patch for patch in images_transform])
#             images = self.transforms(images)
        return images, patient_path.stem, label
    
    def __len__(self):
        return len(self.patient_paths)
    
    def read_random_patches(self, patch_paths):
        random.shuffle(patch_paths)
        patch_paths = patch_paths[:self.num_patches]
        images = []
        for path in patch_paths:
            img = np.array(Image.open(path))
            img = torch.tensor(img).permute(2,0,1)/255.
            images.append(img)
        images = torch.stack([patch for patch in images])
        return images
    
    
class SlideDatasetFromList(Dataset):
    '''
    Dataset Class for Slide-level training (each datapoint is a slide)
    patient_paths: path to patient folder
    num_patches: number of patches to sample from each slide
    transfroms: image augmentation
    '''
    def __init__(self, patient_paths, num_patches, transform=None, class_label=None):
        self.patient_paths = patient_paths
        self.slides = self.get_slide_ids(self.patient_paths)
        self.random_seed = random.sample(range(0, 10000), len(self.slides)) # seed for augmentation
        self.num_patches = num_patches
        self.transform = transform
        self.class_label = class_label # {'nodepos':1, 'nodeneg':0}
        
    def __getitem__(self, idx):
        slide_id = self.slides[idx] # get path to slide folder
        label = self.class_label[slide_id.parent.parents[0].name] # get label from folder path
        images = self.read_random_patches(list(slide_id.glob('*.png')))
        
        # apply transformation to each patch in the stack (revisit to optimize)
        if self.transform:
            images_transform = []
            for patch in images:
                random.seed(self.random_seed[idx])
                torch.manual_seed(self.random_seed[idx])
                images_transform.append(self.transform(patch))
            images = torch.stack([patch for patch in images_transform])
        return images, str(slide_id), label
    
    def __len__(self):
        return len(self.slides)
    
    def get_slide_ids(self, patient_paths):
        '''
        Get path to slides from patient folders
        '''
        slide_ids = []
        for patient_path in patient_paths:
            #FIXME: TEMP CHANGE FOR TESTING PIPELINE & SHOULD BE REVERTED BEFORE PRODUCTION
            import socket
            if 'ntomita' in socket.gethostname():
                patient_path = Path(str(patient_path).replace('/pool2/users/diana', '/databig/POPPSlide'))
            slide_ids.extend(list(patient_path.glob('*')))
        return slide_ids
    
    def read_random_patches(self, patch_paths):
        '''
        Sample random pateches from all available patches. 
        Sample without replacement if number of available patches > num_patches.
        Otherwise sample with replacement.
        '''
        # sample without replacement when #available patches >= num_patchs
        if len(patch_paths) >= self.num_patches: 
            random.shuffle(patch_paths)
            patch_paths = patch_paths[:self.num_patches]
        # sample with replacement when #available patches < num_patchs
        else:     
            patch_paths = np.random.choice(patch_paths,self.num_patches)
        images = []
        for path in patch_paths:
            img = np.array(Image.open(path))
            img = torch.tensor(img).permute(2,0,1)/255. #XXX: WHY LOAD DIRECTLY?
            images.append(img)
        images = torch.stack([patch for patch in images])
        return images
