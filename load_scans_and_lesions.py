#%%
import glob
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label
import matplotlib.patches as patches
from scipy.ndimage import binary_closing, binary_erosion, binary_dilation
from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import nibabel as nib
import scipy

from time import time
import torch.nn.functional as F
from skimage.morphology import remove_small_holes, remove_small_objects
import argparse
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

from utils import (
    select_lesions_match_conditions2, 
    superpixels, 
    coords_min_max_2D, 
    make_list_of_targets_and_seeds)
# from utils_cell_auto import (correct_label_in_plot, 
#     create_sobel_and_identity, 
#     prepare_seed, 
#     epochs_in_inner_loop, 
#     plot_loss_and_lesion_synthesis,
#     to_rgb,
#     CeA_00, CeA_0x, CeA_BASE,
#     )

#%%
class LesionLoader(Dataset):
    def __init__(self, folder_source):
        self.folder_source = folder_source
        files_scan = sorted(glob.glob(os.path.join(self.folder_source,"*.npy")))
        files_mask = sorted(glob.glob(os.path.join(self.folder_source,"*.npz")))
        self.keys = ("image", "label")
        self.files = [{self.keys[0]: img, self.keys[1]: seg} for img, seg in zip(files_scan, files_mask)]
    
    def scale(self, array,old_min=-1000, old_max=500):
        array2 = (array - (old_min))/(old_max - old_min)
        array2 = np.clip(array2, 0, 1)
        return array2

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        scan = np.load(self.files[index]['image'])
        scan = self.scale(scan)
        scan_mask = np.load(self.files[index]['label'])
        scan_mask = (scan_mask.f.arr_0)
        keys = ("image", "label")
        inside_dict = {"filename_or_obj":self.folder_source}
        output = {keys[0]: scan, keys[1]: scan_mask, "image_meta_dict":inside_dict}
        return output

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
# parser = argparse.ArgumentParser(description='scan name ending in _ct/')
# parser.add_argument('--SCAN_NAME', nargs='?')
# parser.add_argument('--only_one_slice', nargs='?', type=int, const=-1)
# parser.add_argument('--BACKGROUND_INTENSITY', nargs='?', type=float, const=0.11)
# parser.add_argument('--STEP_SIZE', nargs='?', type=float, const=1)
# parser.add_argument('--SCALE_MASK', nargs='?', type=float, const=1)
# parser.add_argument('--SEED_VALUE', nargs='?', type=float, const=.19)
# parser.add_argument('--PRETRAIN', nargs='?', type=int, const=100)
# parser.add_argument('--CH0_1', nargs='?', type=int, const=1)
# parser.add_argument('--CH1_16', nargs='?', type=int, const=16)
# parser.add_argument('--ALIVE_THRESH', nargs='?', type=float, const=0.1)
# parser.add_argument('--GROW_ON_K_ITER', nargs='?', type=int, const=1)
# parser.add_argument('--INNER_ITER', nargs='?', type=int, const=1)
# parser.add_argument('--SKIP_LESIONS', nargs='?', type=int, const=0)
# args = parser.parse_args()

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
args = ArgsBaseline('volume-covid19-A-0014', 0)

print(f'\nSCAN_NAME={args.SCAN_NAME}, only_one_slice={args.only_one_slice}, \
BACKGROUND_INTENSITY={args.BACKGROUND_INTENSITY}, STEP_SIZE={args.STEP_SIZE},\
SCALE_MASK={args.SCALE_MASK}, SEED_VALUE={args.SEED_VALUE}, PRETRAIN={args.PRETRAIN}, \
CH0_1={args.CH0_1}, CH1_16={args.CH1_16}, ALIVE_THRESH={args.ALIVE_THRESH}, \
GROW_ON_K_ITER={args.GROW_ON_K_ITER}, INNER_ITER={args.INNER_ITER}, \
SKIP_LESIONS={args.SKIP_LESIONS}')

#%%
# LOAD INDIVIDUAL LESIONS
folder_source = f'/mnt/90cf2a10-3cf8-48a6-9234-9973231cadc6/COVID19/individual_lesions/{args.SCAN_NAME}_ct/'

#%%
#PYTORCH
ds = LesionLoader(folder_source)
loader_lesions = DataLoader(ds, batch_size=1, shuffle=False)

#%%
mask_sizes=[]
cluster_sizes = []
targets_all = []
for idx_mini_batch,mini_batch in enumerate(loader_lesions):
    if idx_mini_batch < args.SKIP_LESIONS:continue #resume incomplete reconstructions

    img = mini_batch['image'].numpy()
    mask = mini_batch['label'].numpy()
    mask = remove_small_objects(mask, 20)
    mask_sizes.append([idx_mini_batch, np.sum(mask)])
    name_prefix = mini_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.npy')[0].split('19-')[-1]
    print(f'mini_batch = {idx_mini_batch} {name_prefix}')
    img_lesion = img*mask
    # if 2nd argument is provided then only analyze that slice
    if args.only_one_slice != -1: 
        slice_used = int(name_prefix.split('_')[-1])
        if slice_used != int(args.only_one_slice): continue

    #%%
    # First use of SLIC, if the lesion is small only need to use SLIC once
    # numSegments = 300 # run slic with large segments to eliminate background & vessels
    SCALAR_LIMIT_CLUSTER_SIZE = 200 #340
    numSegments = np.max([np.sum(mask[0]>0)//SCALAR_LIMIT_CLUSTER_SIZE, 1]) # run slic with large segments to eliminate background & vessels
    TRESH_BACK = 0.10 #orig=0.15
    THRES_VESSEL = 0.7 #orig=.5
    if numSegments>1: # if mask is large then superpixels
        SCALAR_SIZE2 = 300
        numSegments = np.max([np.sum(mask[0]>0)//SCALAR_SIZE2, 4])
        segments = slic((img[0]).astype('double'), n_segments = numSegments, mask=mask[0], sigma = .2, multichannel=False, compactness=.1)
        background, lesion_area, vessels = superpixels((img[0]).astype('double'), segments, background_threshold=TRESH_BACK, vessel_threshold=THRES_VESSEL)
        mask_slic = lesion_area>0
        boundaries = mark_boundaries(mask_slic*img[0], segments)[...,0]
        # label_seg, nr_seg = label(segments)
    else: # small lesion (use the original mask)
        numSegments=-1
        background, lesion_area, vessels = superpixels((img[0]).astype('double'), mask[0], background_threshold=TRESH_BACK, vessel_threshold=THRES_VESSEL)
        mask_slic = mask[0]
        boundaries = np.zeros_like(mask_slic)
        segments = mask[0]
    segments_sizes = [np.sum(segments==i_segments) for i_segments in np.unique(segments)[1:]]
    cluster_sizes.append(segments_sizes)
    segments_sizes = [str(f'{i_segments}') for i_segments in segments_sizes]
    segments_sizes = '\n'.join(segments_sizes)

    # save vars for fig_slic
    
    background_plot = background;  lesion_area_plot = lesion_area
    vessels_plot = vessels; boundaries_plot = boundaries
    
    labelled, nr = label(mask_slic)
    mask_dil = remove_small_holes(remove_small_objects(mask_slic, 50))
    labelled2, nr2 = label(mask_dil)


    #%%
    tgt_minis, tgt_minis_coords, tgt_minis_masks, tgt_minis_big, tgt_minis_coords_big, tgt_minis_masks_big = select_lesions_match_conditions2(segments, img[0], skip_index=0)
    targets, coords, masks, seeds = make_list_of_targets_and_seeds(tgt_minis, tgt_minis_coords, tgt_minis_masks, seed_value=args.SEED_VALUE, seed_method='max')
    targets_all.append(len(targets))

#%%
#SCAN LOADER
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
# %%
