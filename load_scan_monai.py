#%%
import glob
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar

import monai
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler
from monai.transforms import (
    AddChanneld, AsDiscreted, CastToTyped, LoadImaged,
    Orientationd, RandAffined, RandCropByPosNegLabeld,
    RandFlipd, RandGaussianNoised, ScaleIntensityRanged,
    Spacingd, SpatialPadd, ToTensord,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.handlers import MetricsSaver
from scipy.ndimage import label
import matplotlib.patches as patches
from scipy.ndimage import binary_closing, binary_erosion, binary_dilation
from pathlib import Path
# %%
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import nibabel as nib
import scipy
#%%
def resample(scan, old_spacing, new_spacing = (1.25, 1.25, 5.0)):
    '''resample to new spacing. From:
    https://www.kaggle.com/allunia/pulmonary-dicom-preprocessing'''
    # spacing = scan_info.header['pixdim'][1:4]
    resize_factor = old_spacing/new_spacing
    new_shape = np.ceil(np.shape(scan) * resize_factor)
    rounded_resize_factor = new_shape / np.shape(scan)
    rounded_new_spacing = old_spacing / rounded_resize_factor
    image = scipy.ndimage.interpolation.zoom(scan, rounded_resize_factor, mode='nearest')
    return image, rounded_new_spacing

def rescale(array, old_min=-1000, old_max=500):
    array2 = (array - (old_min))/(old_max - old_min)
    array2 = np.clip(array2, 0, 1)
    return array2
#%%    
def get_xforms_load_scans(mode="load", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "load":
        dtype = (np.float32, np.uint8)
    # xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
    xforms.extend([CastToTyped(keys, dtype=dtype)])
    return monai.transforms.Compose(xforms)

# def get_xforms_load(mode="load", keys=("image", "label")):
#     """returns a composed transform for train/val/infer."""

#     xforms = [
#         LoadImaged(keys),
#         AddChanneld(keys),
#         Orientationd(keys, axcodes="LPS"),
#         Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
#         ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
#     ]
#     if mode == "load":
#         dtype = (np.float32, np.uint8)
#     # xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
#     xforms.extend([CastToTyped(keys, dtype=dtype)])
#     return monai.transforms.Compose(xforms)


#%%
class ArgsBaseline():   
    def __init__(self, SCAN_NAME, SKIP_LESIONS):
        self.SCAN_NAME=  SCAN_NAME
        self.only_one_slice = -1 
        self.BACKGROUND_INTENSITY = -1 
        self.STEP_SIZE = -1 
        self.SCALE_MASK = -1 
        self.SEED_VALUE = 1 
        self.PRETRAIN = 0 
        self.CH0_1 = -1 
        self.CH1_16 = -1 
        self.ALIVE_THRESH = .1 
        self.GROW_ON_K_ITER = 1 
        self.INNER_ITER = 1 
        self.SKIP_LESIONS = SKIP_LESIONS # WARNING SHOULD BE 0
args = ArgsBaseline('volume-covid19-A-0014', 34)

# %%
# LOAD ORIGINAL SCAN /=== FOR fig_slic, later can be removed
data_folder = '/mnt/90cf2a10-3cf8-48a6-9234-9973231cadc6/COVID19/COVID-19-20/Train'
images= [f'{data_folder}/{args.SCAN_NAME}_ct.nii.gz']
labels= [f'{data_folder}/{args.SCAN_NAME}_seg.nii.gz']
keys = ("image", "label")
files_scans = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images, labels)]

batch_size = 1
transforms_load = get_xforms_load_scans("load", keys)
ds_scans = monai.data.CacheDataset(data=files_scans, transform=transforms_load)
loader_scans = monai.data.DataLoader(
        ds_scans,
        batch_size=batch_size,
        shuffle=False, #should be true for training
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
#%%
for idx_mini_batch, mini_batch in enumerate(loader_scans):
    # if idx_mini_batch==1:break #OMM
    BATCH_IDX=0
    scan = mini_batch['image'][BATCH_IDX][0,...]
    scan_mask = mini_batch['label'][BATCH_IDX][0,...]
    scan_name = mini_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.nii')[0][:-3]
print(f'working on scan= {scan_name}')
assert scan_name == args.SCAN_NAME, 'cannot load that scan'
scan = scan.numpy()   #ONLY READ ONE SCAN (WITH PREVIOUS BREAK)
scan_mask = scan_mask.numpy()

# %%
SLICE = 34
print(np.shape(scan),np.shape(scan_mask))
fig, ax = plt.subplots(1,2,figsize=(12,4))
ax[0].imshow(scan[...,SLICE])
ax[1].hist(scan[...,SLICE].flatten());
scan_monai = scan 




#================ USINH PYTORCH
# %%
class CTLoader(Dataset):
    def __init__(self, path_data: Path, scan_name: str, transforms):
        self.keys = ("image", "label")
        self.path_data = path_data
        self.transform = transforms
        self.scan_name = scan_name
        images= [self.path_data / f'{self.scan_name}_ct.nii.gz' ]
        labels= [self.path_data / f'{self.scan_name}_seg.nii.gz' ]
        self.files_scans = [{self.keys[0]: img, self.keys[1]: seg} for img, seg in zip(images, labels)]
    
    def __len__(self) -> int:
        return len(self.files_scans)

    def __getitem__(self, index: int):
        scan = nib.load(self.files_scans[0]['image']).get_fdata()
        scan_mask = nib.load(self.files_scans[0]['label']).get_fdata()
        scan_info = nib.load(self.files_scans[0]['image'])
        scan_pixdim = scan_info.header['pixdim'][1:4]
        print(scan_info)
        return scan, scan_mask, scan_pixdim

#%%
path_source = Path('/mnt/90cf2a10-3cf8-48a6-9234-9973231cadc6/COVID19/COVID-19-20/Train/')
scan_dataset = CTLoader(path_source, args.SCAN_NAME, None)
scan_dataloader = DataLoader(scan_dataset, batch_size=1, shuffle=False)
#%%
scan_, scan_mask_, scan_pixdim = next(iter(scan_dataloader))        
scan_ = np.squeeze(scan_.numpy())
scan_mask_ = np.squeeze(scan_mask_.numpy())
scan_pixdim = np.squeeze(scan_pixdim.numpy())
print(np.shape(scan_), np.shape(scan_mask_))


#%%
SLICE=34
scan, rounded_new_spacing = resample(scan_, scan_pixdim)
scan_mask, rounded_new_spacing = resample(scan_mask_, scan_pixdim)
scan = rescale(scan)
print(np.shape(scan),np.shape(scan_mask))
fig, ax = plt.subplots(1,2,figsize=(12,4))
ax[0].imshow(scan[...,SLICE])
# ax[0].imshow(scan_mask[...,SLICE], alpha=.3)
ax[1].hist(scan[...,SLICE].flatten());

#%%
print(np.shape(scan_monai),np.shape(scan_mask))
fig, ax = plt.subplots(1,2,figsize=(12,4))
ax[0].imshow(scan_monai[...,SLICE])
# ax[0].imshow(scan_mask[...,SLICE], alpha=.3)
ax[1].hist(scan_monai[...,SLICE].flatten());

#%%
error = np.abs(scan_monai-scan)
plt.imshow(error[...,SLICE], vmin=0, vmax=1)
plt.colorbar()
np.sum(np.abs(error[...,SLICE]))



#============= EXTRA
# %%
# images = ['/mnt/90cf2a10-3cf8-48a6-9234-9973231cadc6/COVID19/COVID-19-20/Train/volume-covid19-A-0014_ct.nii.gz']
# labels = ['/mnt/90cf2a10-3cf8-48a6-9234-9973231cadc6/COVID19/COVID-19-20/Train/volume-covid19-A-0014_seg.nii.gz']
# keys = ("image", "label")
# files_scans = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images, labels)]
# # %%
# files_scans
# # %%
# scan_info = nib.load(files_scans[0]['image'])
# scan_nii = scan_info.get_fdata()
# scan_mask_nii = nib.load(files_scans[0]['label']).get_fdata()
# np.shape(scan_nii), np.shape(scan_mask_nii)

# #%%
# SLICE=34
# scan_nii2, new_spacing = resample(scan_nii, scan_info.header['pixdim'][1:4])
# print(np.shape(scan_nii2))
# fig, ax = plt.subplots(1,2,figsize=(12,4))
# ax[0].imshow(scan_nii2[...,SLICE])
# ax[1].hist(scan_nii2[...,SLICE].flatten());
# # %%
# fig, ax = plt.subplots(1,2,figsize=(12,4))
# ax[0].imshow(scan_nib[...,SLICE])
# ax[1].hist(scan_nib[...,SLICE].flatten());
# # %%
# scan_nib_all = nib.load(files_scans[0]['image'])
# scan_nib_all
# # %%
# print(scan_nib_all.header)
# # %%
# scan_nib_all.header['pixdim'][1:4]
# # %%
# spacing = scan_nib_all.header['pixdim'][1:4]
# new_spacing = (1.25, 1.25, 5.0)
# resize_factor = spacing/new_spacing
# print(resize_factor)
# print(np.shape(scan_nib))
# new_shape = np.ceil(np.shape(scan_nib) * resize_factor)
# print(new_shape, np.shape(scan_monai))
# # %%
# rounded_resize_factor = new_shape / np.shape(scan_nib)
# rounded_new_spacing = spacing / rounded_resize_factor
# # %%
# image = scipy.ndimage.interpolation.zoom(scan_nib, rounded_resize_factor, mode='nearest')
# print(np.shape(scan_nib), np.shape(image))
# # %%
# np.shape(image), np.shape(scan_monai)
# # %%
# SLICE = 34
# print(np.shape(image))
# fig, ax = plt.subplots(1,2,figsize=(12,4))
# ax[0].imshow(image[...,SLICE])
# ax[1].hist(image[...,SLICE].flatten());

