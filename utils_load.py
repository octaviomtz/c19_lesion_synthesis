import numpy as np
import monai
import torch
import glob
import os
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from scipy.ndimage import label
from skimage.morphology import remove_small_holes, remove_small_objects
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandFlipd,
    CastToTyped,
)
from utils import (
    superpixels,
    make_list_of_targets_and_seeds,
    select_lesions_match_conditions2,
)

# FUNCTIONS
def get_xforms_scans_or_synthetic_lesions(mode="scans", keys=("image", "label")):
    """returns a composed transform for scans or synthetic lesions."""
    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
    ]
    dtype = (np.int16, np.uint8)
    if mode == "synthetic":
        xforms.extend([
          ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
        ])
        dtype = (np.float32, np.uint8)
    xforms.extend([CastToTyped(keys, dtype=dtype)])
    return monai.transforms.Compose(xforms)

def get_xforms_load(mode="load", keys=("image", "label")):
    """returns a composed transform."""
    xforms = [
        LoadImaged(keys),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "load":
        dtype = (np.float32, np.uint8)
    xforms.extend([CastToTyped(keys, dtype=dtype)])
    return monai.transforms.Compose(xforms)

def load_synthetic_lesions_scans_and__individual_lesions(SCAN_NAME, data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20_v2/Train', batch_size=1):
    '''load three components
    - the original scans
    - the synthetic lesions
    - the original lesions
    - the pseudo-healthy texture'''
    images= [f'{data_folder}/{SCAN_NAME}_ct.nii.gz']
    labels= [f'{data_folder}/{SCAN_NAME}_seg.nii.gz']
    keys = ("image", "label")
    files_scans = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images, labels)]
    
    transforms_load = get_xforms_scans_or_synthetic_lesions("synthetic", keys)
    ds_synthetic = monai.data.CacheDataset(data=files_scans, transform=transforms_load)
    loader_synthetic = monai.data.DataLoader(
            ds_synthetic,
            batch_size=batch_size,
            shuffle=False, #should be true for training
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
    for idx_mini_batch, mini_batch in enumerate(loader_synthetic):
        # if idx_mini_batch==6:break #OMM
        BATCH_IDX=0
        scan_synthetic = mini_batch['image'][BATCH_IDX][0,...].numpy()
        scan_mask = mini_batch['label'][BATCH_IDX][0,...].numpy()
        name_prefix = mini_batch['image_meta_dict']['filename_or_obj'][0].split('Train/')[-1].split('.nii')[0]
        sum_TEMP_DELETE = np.sum(scan_mask)
        print(name_prefix)

    # LOAD SCANS
    transforms_load = get_xforms_scans_or_synthetic_lesions("scans", keys)
    ds_scans = monai.data.CacheDataset(data=files_scans, transform=transforms_load)
    loader_scans = monai.data.DataLoader(
            ds_scans,
            batch_size=batch_size,
            shuffle=False, #should be true for training
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    for idx_mini_batch, mini_batch in enumerate(loader_scans):
        # if idx_mini_batch==1:break #OMM
        BATCH_IDX=0
        scan = mini_batch['image'][BATCH_IDX][0,...]
        scan_mask = mini_batch['label'][BATCH_IDX][0,...]
        scan_name = mini_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.nii')[0][:-3]
    print(f'working on scan= {scan_name}')
    assert scan_name == SCAN_NAME, 'cannot load that scan'
    scan = scan.numpy()   #ONLY READ ONE SCAN (WITH PREVIOUS BREAK)
    scan_mask = scan_mask.numpy()

    # LOAD INDIVIDUAL LESIONS
    folder_source = f'/content/drive/MyDrive/Datasets/covid19/COVID-19-20/individual_lesions/{SCAN_NAME}_ct/'
    files_scan = sorted(glob.glob(os.path.join(folder_source,"*.npy")))
    files_mask = sorted(glob.glob(os.path.join(folder_source,"*.npz")))
    keys = ("image", "label")
    files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(files_scan, files_mask)]
    print(len(files_scan), len(files_mask), len(files))
    transforms_load = get_xforms_load("load", keys)
    ds_lesions = monai.data.CacheDataset(data=files, transform=transforms_load)
    loader_lesions = monai.data.DataLoader(
            ds_lesions,
            batch_size=batch_size,
            shuffle=False, #should be true for training
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    # LOAD SYTHETIC INPAINTED PSEUDO-HEALTHY TEXTURE
    path_synthesis_old = '/content/drive/My Drive/Datasets/covid19/results/cea_synthesis/patient0/'
    texture_orig = np.load(f'{path_synthesis_old}texture.npy.npz')
    texture_orig = texture_orig.f.arr_0
    texture = texture_orig + np.abs(np.min(texture_orig))# + .07

    return scan, scan_mask, loader_lesions, texture

def superpixels_applied(loader_lesions, ONLY_ONE_SLICE, TRESH_PLOT=20, SKIP_LESIONS=0, SEED_VALUE=1): 
    mask_sizes = []
    cluster_sizes = []
    targets_all = []
    flag_only_one_slice = False

    # img = mask_slic = boundaries_plot = segments = segments_sizes = coords_big = idx_mini_batch = numSegments = np.random.rand(10,10)
    # return  img, mask_slic, boundaries_plot, segments, segments_sizes, coords_big, idx_mini_batch, numSegments

    for idx_mini_batch, mini_batch in enumerate(loader_lesions):
        if idx_mini_batch < SKIP_LESIONS:continue #resume incomplete reconstructions

        img = mini_batch['image'].numpy()
        mask = mini_batch['label'].numpy()
        mask = remove_small_objects(mask, 20)
        mask_sizes.append([idx_mini_batch, np.sum(mask)])
        name_prefix = mini_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.npy')[0].split('19-')[-1]
        img_lesion = img*mask

        # if 2nd argument is provided then only analyze that slice
        if ONLY_ONE_SLICE != -1: 
            slice_used = int(name_prefix.split('_')[-1])
            if slice_used != int(ONLY_ONE_SLICE): continue
            else: flag_only_one_slice = True

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

        tgt_minis, tgt_minis_coords, tgt_minis_masks, tgt_minis_big, tgt_minis_coords_big, tgt_minis_masks_big = select_lesions_match_conditions2(segments, img[0], skip_index=0)
        targets, coords, masks, seeds = make_list_of_targets_and_seeds(tgt_minis, tgt_minis_coords, tgt_minis_masks, seed_value=SEED_VALUE, seed_method='max')
        targets_all.append(len(targets))

        coords_big = name_prefix.split('_')
        coords_big = [int(i) for i in coords_big[1:]]
        
        if flag_only_one_slice: break
    return  img, mask_slic, boundaries_plot, segments, segments_sizes, coords_big, idx_mini_batch, numSegments
        