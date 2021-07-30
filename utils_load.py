import numpy as np
import monai
import torch
import glob
import os
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from scipy.ndimage import label
from skimage.morphology import remove_small_holes, remove_small_objects
from PIL import Image
from scipy.ndimage.morphology import binary_erosion
import matplotlib.pyplot as plt
import streamlit as st
from copy import copy
from skimage.restoration import inpaint
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
from utils_replace_lesions import (
    read_cea_aug_slice2,
    pseudo_healthy_with_texture,
    get_decreasing_sequence,
    get_orig_scan_in_lesion_coords,
    make_mask_ring,
    normalize_new_range4,
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

def load_only_original_scans(SCAN_NAME, data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20_v2/Train', batch_size=1):
    '''load only one components
    - the original scans (and their masks)'''
    images= [f'{data_folder}/{SCAN_NAME}_ct.nii.gz']
    labels= [f'{data_folder}/{SCAN_NAME}_seg.nii.gz']
    keys = ("image", "label")
    files_scans = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images, labels)]
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
    assert scan_name == SCAN_NAME, 'cannot load that scan'
    scan = scan.numpy()   #ONLY READ ONE SCAN (WITH PREVIOUS BREAK)
    scan_mask = scan_mask.numpy()
    return scan, scan_mask

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

def superpixels2(img, mask):
    mask = remove_small_objects(mask, 20)
    SCALAR_LIMIT_CLUSTER_SIZE = 200 #340
    numSegments = np.max([np.sum(mask > 0)//SCALAR_LIMIT_CLUSTER_SIZE, 1]) # run slic with large segments to eliminate background & vessels
    TRESH_BACK = 0.10 
    THRES_VESSEL = 0.7 
    print(numSegments)
    if numSegments>1: # if mask is large then superpixels
        SCALAR_SIZE2 = 300
        numSegments = np.max([np.sum(mask > 0)//SCALAR_SIZE2, 4])
        segments = slic((img/255).astype('double'), n_segments = numSegments, mask=mask, sigma = .2, multichannel=False, compactness=.1)
        print(f'img={np.min(img),np.max(img)}')
        print(f'segments={np.sum(segments)}, {np.unique(segments)}')
        print(f'img={np.shape(img)}, mask={np.shape(mask)}')
        background, lesion_area, vessels = superpixels((img).astype('double'), segments, background_threshold=TRESH_BACK, vessel_threshold=THRES_VESSEL)
        mask_slic = lesion_area>0
        boundaries = mark_boundaries(mask_slic*img, segments)[...,0]
        # label_seg, nr_seg = label(segments)
    else: # small lesion (use the original mask)
        numSegments=-1
        background, lesion_area, vessels = superpixels((img).astype('double'), mask, background_threshold=TRESH_BACK, vessel_threshold=THRES_VESSEL)
        mask_slic = mask
        boundaries = np.zeros_like(mask_slic)
        segments = mask
    return mask_slic, boundaries, segments

def from_scan_to_3channel(img, slice=34, normalize=True, rotate=0):
    """
    Transform from hounsfield units to a normalized (0-255)
    image that streamlit can plot
    Args:
        img (numpy array): [description]
        slice (int, optional): [description]. Defaults to 34.
        normalize (bool, optional): [description]. Defaults to True.
        rotate (int, optional): [description]. Defaults to 0.

    Returns:
        img [int]: Image ready to plot
    """
    if normalize:
        img = normalize_scan(img)
    if rotate:
        img = np.rot90(img,-1)
    img = np.expand_dims(img,-1)
    img = np.repeat(img,3,-1)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img

def from_scan_to_3channel2(img, slice=34, normalize=True, rotate=0):
    img = img[...,slice]
    if normalize:
        img = normalize_scan(img)
    if rotate:
        img = np.rot90(img,-1)
    fig = plt.figure()
    # plt.imshow(img)    
    img = (img*255).astype(np.uint8)
    img = Image.fromarray(img).convert('RGB')
    return img

def normalize_scan(image, MIN_BOUND=-1000, MAX_BOUND=500):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def scale_rect_coords_and_compare_nodule_coords(scan, rect_coords, coords_nodule, CANVA_HEIGHT=400, CANVA_WIDTH=400, THRESH=10):
    """
    Scale the coords of the square entered by the user using the 
    canva size and the shape of the original image
    Args:
        scan(numpy array): 2D scan
        rect_coords(dict): coordinated entered by the user
    Returns
        coords_scales(list[float]): scales coordinated that match the location in the original dataset
        dist_coords(float): euclidian distance between the coords
        dist_coords_bool: True if dist_coords is smaller than THRESH
    """
    dist_coords_bool = False
    coord_x1 = rect_coords.get('left'); 
    coord_x2 = rect_coords.get('left') + rect_coords.get('width') 
    coord_y1 = rect_coords.get('top'); 
    coord_y2 =  rect_coords.get('top') + rect_coords.get('height')
    canva_shape0, canva_shape1 = np.shape(scan)
    scale_y = canva_shape0/CANVA_HEIGHT
    scale_x = canva_shape1/CANVA_WIDTH
    coords_scaled = coord_y1*scale_y, coord_y2*scale_y, coord_x1*scale_x, coord_x2*scale_x
    # calculate euclidian distance
    dist_coords = np.linalg.norm(np.asarray(coords_scaled)-np.asarray(coords_nodule))
    if dist_coords < THRESH:
        dist_coords_bool = True
    return coords_scaled, dist_coords, dist_coords_bool

def figures_zoom_and_superpixels(scan, mask, coords, boundaries, offset = 5):
    '''
    Create figure handles of the (1) whole scan, (2) zoom to lesion
        and (3) zoom to lesion with superpixels boundaries included
    Args:
        scan(numpy array):
        mask(numpy array[bool]):
    Return:
        Three Matplotlib figure handles
    '''
    y1, y2, x1, x2 = coords
    y1, y2, x1, x2 = y1-offset, y2+offset, x1-offset, x2+offset
    mask_eroded = binary_erosion(mask)
    mask_boundary = mask - mask_eroded
    fig_zoom1 = plt.figure()
    plt.imshow(scan, cmap='gray')
    plt.imshow(mask, cmap='flag_r', alpha=.3)
    plt.axis('off')
    fig_zoom2 = plt.figure()
    plt.imshow(scan[y1: y2, x1: x2], cmap='gray')
    # plt.imshow(mask_boundary[y1: y2, x1: x2], cmap='flag_r', alpha=.3)
    plt.axis('off')
    fig_zoom3 = plt.figure()
    plt.imshow(scan[y1: y2, x1: x2], cmap='gray')
    plt.imshow(boundaries[y1: y2, x1: x2], cmap='flag_r', alpha=.3)
    plt.axis('off')
    return fig_zoom1, fig_zoom2, fig_zoom3

@st.cache(suppress_st_warning=True)
def load_synthetic_texture(path_synthesis = '/content/drive/My Drive/Datasets/covid19/results/cea_synthesis/patient0/'):
    texture_orig = np.load(f'{path_synthesis}texture.npy.npz')
    texture_orig = texture_orig.f.arr_0
    texture = texture_orig + np.abs(np.min(texture_orig))# + .07
    return texture

def replace_with_nCA(scan, SCAN_NAME, SLICE, texture, mask_outer_ring = False, POST_PROCESS = True, blur_lesion = False, TRESH_PLOT=10):
    scan_slice = scan/255
    path_parent = '/content/drive/My Drive/Datasets/covid19/COVID-19-20_augs_cea/'
    path_synthesis_ = f'{path_parent}CeA_BASE_grow=1_bg=-1.00_step=-1.0_scale=-1.0_seed=1.0_ch0_1=-1_ch1_16=-1_ali_thr=0.1/'
    path_synthesis = f'{path_synthesis_}{SCAN_NAME}/'
    lesions_all, coords_all, masks_all, names_all, loss_all = read_cea_aug_slice2(path_synthesis, SLICE=SLICE)
    V_MAX = np.max(scan_slice)
    slice_healthy_inpain = pseudo_healthy_with_texture(scan_slice, lesions_all, coords_all, masks_all, names_all, texture)
    decreasing_sequence = get_decreasing_sequence(255, splits= 20) 
    arrays_sequence = []
    images=[]
    mse_gen = []
    for GEN in decreasing_sequence:

        slice_healthy_inpain2 = copy(slice_healthy_inpain)
        synthetic_intensities=[]
        mask_for_inpain = np.zeros_like(slice_healthy_inpain2)
        mse_lesions = []
        for idx_x, (lesion, coord, mask, name) in enumerate(zip(lesions_all, coords_all, masks_all, names_all)):
            #get the right coordinates
            coords_big2 = [int(i) for i in name.split('_')[1:5]]
            coords_sums = coord + coords_big2
            new_coords_mask = np.where(mask==1)[0]+coords_sums[0], np.where(mask==1)[1]+coords_sums[2]
            if GEN<60:
                if POST_PROCESS:
                    syn_norm = normalize_new_range4(lesion[GEN], scan_slice[new_coords_mask], scale=.5)
                else:
                    syn_norm = lesion[GEN]
            else:
                syn_norm = lesion[GEN]
            # get the MSE between synthetic and original
            orig_lesion = get_orig_scan_in_lesion_coords(scan_slice, new_coords_mask)
            mse_lesions.append(np.mean(mask*(syn_norm - orig_lesion)**2))

            syn_norm = syn_norm * mask
            if blur_lesion:
                syn_norm = blur_masked_image(syn_norm, kernel_blur=(2,2))
            # add cea syn with absolute coords
            new_coords = np.where(syn_norm>0)[0]+coords_sums[0], np.where(syn_norm>0)[1]+coords_sums[2]
            slice_healthy_inpain2[new_coords] = syn_norm[syn_norm>0]

            synthetic_intensities.extend(syn_norm[syn_norm>0])

            # inpaint the outer ring
            if mask_outer_ring:
                mask_ring = make_mask_ring(syn_norm>0)
                new_coords_mask_inpain = np.where(mask_ring==1)[0]+coords_sums[0], np.where(mask_ring==1)[1]+coords_sums[2] # mask outer rings for inpaint
                mask_for_inpain[new_coords_mask_inpain] = 1
        
        mse_gen.append(mse_lesions)
        if mask_outer_ring:
            slice_healthy_inpain2 = inpaint.inpaint_biharmonic(slice_healthy_inpain2, mask_for_inpain)

        arrays_sequence.append(slice_healthy_inpain2[coords_big2[0]-TRESH_PLOT:coords_big2[1]+TRESH_PLOT,coords_big2[2]-TRESH_PLOT:coords_big2[3]+TRESH_PLOT])
        images.append(slice_healthy_inpain2[coords_big2[0]-TRESH_PLOT:coords_big2[1]+TRESH_PLOT,coords_big2[2]-TRESH_PLOT:coords_big2[3]+TRESH_PLOT])

    return arrays_sequence, images, decreasing_sequence