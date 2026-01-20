import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples,to_pixel_samples_gray


@register('sr-implicit-paired-he-ihc')
class SRImplicitPairedHEIHC(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        
        crop_lr = img_lr
        crop_hr = img_hr

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'gt': hr_rgb
        }


@register('sr-implicit-paired-he-ihc-downsample')
class SRImplicitPairedHEIHCDownsample(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_he_lr, img_ihc_lr,img_he_hr,img_ihc_hr = self.dataset[idx]
        

        lr_coord, lr_rgb = to_pixel_samples(img_ihc_lr.contiguous())
        hr_coord, hr_rgb = to_pixel_samples(img_ihc_hr.contiguous())

        return {
            'inp': img_he_lr,
            'coord': lr_coord,
            'gt': lr_rgb,
            'inp1': img_he_hr,
            'coord1': hr_coord,
            'gt1': hr_rgb,
        }

