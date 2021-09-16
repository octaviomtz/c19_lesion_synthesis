#pip3 install -r requirements.txt

# %%
import os
import numpy as np
import monai
import torch
from copy import copy
from tqdm import tqdm
from skimage.restoration import inpaint
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import glob
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.morphology import remove_small_holes, remove_small_objects
from scipy.ndimage import label
import imageio
from time import time
import torch.nn.functional as F
from utils_replace_lesions import (
    read_cea_aug_slice2,
    pseudo_healthy_with_texture,
    get_decreasing_sequence,
    normalize_new_range4,
    get_orig_scan_in_lesion_coords,
    blur_masked_image,
    make_mask_ring
)
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
    RandFlipd,
    RandFlipd,
    CastToTyped,
)
from utils import (
    superpixels,
    make_list_of_targets_and_seeds,
    fig_superpixels_only_lesions,
    select_lesions_match_conditions2,
)
from utils_replace_lesions import (
    fig_blend_lesion,
)
from utils_cell_auto import (correct_label_in_plot, 
    create_sobel_and_identity, 
    prepare_seed, 
    epochs_in_inner_loop, 
    plot_loss_and_lesion_synthesis,
    to_rgb,
    CeA_BASE, CeA_BASE_1CNN,
    )

#%% FUNCTIONS
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

#%%
def boundaries_superpixels(img, mask):
    mask = remove_small_objects(mask, 20)
    SCALAR_LIMIT_CLUSTER_SIZE = 200 #340
    numSegments = np.max([np.sum(mask > 0)//SCALAR_LIMIT_CLUSTER_SIZE, 1]) # run slic with large segments to eliminate background & vessels
    TRESH_BACK = 0.10 
    THRES_VESSEL = 0.7 
    print(numSegments)
    if numSegments>1: # if mask is large then superpixels
        SCALAR_SIZE2 = 300
        numSegments = np.max([np.sum(mask > 0)//SCALAR_SIZE2, 4])
        segments = slic((img).astype('double'), n_segments = numSegments, mask=mask, sigma = .2, multichannel=False, compactness=.1)
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
    return mask_slic, boundaries, segments, numSegments

def how_large_is_each_segment(segments):
    '''return how many pixels in each segment '''
    segments_sizes = [np.sum(segments==i_segments) for i_segments in np.unique(segments)[1:]]
    segments_sizes = [str(f'{i_segments}') for i_segments in segments_sizes]
    segments_sizes = '\n'.join(segments_sizes)
    return segments_sizes


#%% LOAD SYNTHETIC LESIONS AND ORIGINAL SCANS

SCAN_NAME = 'volume-covid19-A-0014'
SLICE= 34 
SKIP_LESIONS = 0
ONLY_ONE_SLICE = 34
GROW_ON_K_ITER = 1 # apply get_alive_mask every GROW_ON_K_ITER
BACKGROUND_INTENSITY = 0.11
STEP_SIZE = 1
SCALE_MASK = 0.19
SEED_VALUE = 0.19
PRETRAIN = 100
CH0_1 = 1
CH1_16 = 15
ALIVE_THRESH = 0.1
ITER_INNER = 60
ITER_VAR = 1
data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20_v2/Train'
images= [f'{data_folder}/{SCAN_NAME}_ct.nii.gz']
labels= [f'{data_folder}/{SCAN_NAME}_seg.nii.gz']
keys = ("image", "label")
files_scans = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images, labels)]
batch_size = 1

#%% nCA HYPERPARAMETERS
device = 'cuda'
num_channels = 16
epochs = 2500
sample_size = 8

# %% LOAD SYNTHETIC LESIONS
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

# %% LOAD SCANS
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

# %% LOAD INDIVIDUAL LESIONS
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

#%% SUPERPIXELS
mask_sizes=[]
cluster_sizes = []
targets_all = []
flag_only_one_slice = False
for idx_mini_batch,mini_batch in enumerate(loader_lesions):
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

    mask_slic, boundaries, segments, numSegments = boundaries_superpixels(img[0], mask[0])
    segments_sizes = how_large_is_each_segment(segments)

    print(f'img = {np.shape(img)}')

    tgt_minis, tgt_minis_coords, tgt_minis_masks, tgt_minis_big, tgt_minis_coords_big, tgt_minis_masks_big = select_lesions_match_conditions2(segments, img[0], skip_index=0)
    targets, coords, masks, seeds = make_list_of_targets_and_seeds(tgt_minis, tgt_minis_coords, tgt_minis_masks, seed_value=SEED_VALUE, seed_method='max')
    targets_all.append(len(targets))

    coords_big = name_prefix.split('_')
    coords_big = [int(i) for i in coords_big[1:]]
    TRESH_PLOT=20
    device = 'cuda'
    num_channels = 16
    epochs = 2500
    sample_size = 8
    path_parent = 'interactive/'
    path_fig = f'{path_parent}CA={GROW_ON_K_ITER}_bg={BACKGROUND_INTENSITY:.02f}_step={STEP_SIZE}_scale={SCALE_MASK}_seed={SEED_VALUE}_ch0_1={CH0_1}_ch1_16={CH1_16}_ali_thr={ALIVE_THRESH}_iter={ITER_INNER}/'
    path_fig = f'{path_fig}{SCAN_NAME}/'
    Path(path_fig).mkdir(parents=True, exist_ok=True)
    fig_superpixels_only_lesions(path_fig, name_prefix, scan, scan_mask, img, mask_slic, boundaries, segments, segments_sizes, coords_big, TRESH_PLOT, idx_mini_batch, numSegments)
    if flag_only_one_slice: break

#%%
tgt_minis, tgt_minis_coords, tgt_minis_masks, tgt_minis_big, tgt_minis_coords_big, tgt_minis_masks_big = select_lesions_match_conditions2(segments, img[0], skip_index=0)
targets, coords, masks, seeds = make_list_of_targets_and_seeds(tgt_minis, tgt_minis_coords, tgt_minis_masks, seed_value=SEED_VALUE, seed_method='max')
targets_all.append(len(targets))

coords_big = name_prefix.split('_')
coords_big = [int(i) for i in coords_big[1:]]
TRESH_P=10

# %% CREATE DEST FOLDERS AND SAVE FIGURE
path_parent = '/content/drive/My Drive/Datasets/covid19/COVID-19-20_augs_cea/CeA_BASE1'
path_save_synthesis = f'{path_parent}_grow={GROW_ON_K_ITER}_bg={BACKGROUND_INTENSITY:.02f}_step={STEP_SIZE}_scale={SCALE_MASK}_seed={SEED_VALUE}_ch0_1={CH0_1}_ch1_16={CH1_16}_ali_thr={ALIVE_THRESH}_iter={ITER_INNER}/'
path_save_synthesis = f'{path_save_synthesis}{SCAN_NAME}/'
Path(path_save_synthesis).mkdir(parents=True, exist_ok=True)
path_synthesis_figs = f'{path_save_synthesis}fig_slic/'
Path(f'{path_synthesis_figs}').mkdir(parents=True, exist_ok=True)
fig_superpixels_only_lesions(path_synthesis_figs, name_prefix, scan, scan_mask, img, mask_slic, boundaries, segments, segments_sizes, coords_big, TRESH_P, idx_mini_batch, numSegments)

#%%
# targets2 = []
# for targ in targets:
#     targ[...,1] = targ[...,1]*.11
#     targets2.append(targ)
# targets = targets2

#%% FIGURE SEEDS
print(np.shape(targets[0]))
fig, ax = plt.subplots(2,2)
for idx, (t,s) in enumerate(zip(targets,seeds)):
    print(f'target={np.shape(t)}{np.unique(t[...,1])} seed={np.shape(s)}{np.unique(s)}')
    ax.flat[idx].imshow(t[...,1])
    ax.flat[idx].imshow(s, alpha=.3)

# %% CELLULAR AUTOMATA
losses_per_lesion = []
for idx_lesion, (target, coord, mask, this_seed) in enumerate(zip(targets, coords, masks, seeds)):
    # if args.only_one_slice: # if 2nd argument is provided then only analyze that slice
    #     print(coord, int(args.only_one_slice))
    #     if idx_lesion != int(args.only_one_slice): continue
    print(f'==== LESION {idx_mini_batch}/{len(ds_lesions)} CLUSTER {idx_lesion}/{len(coords)}. {name_prefix}')
    # prepare seed
    seed, seed_tensor, seed_pool = prepare_seed(target, this_seed, device, num_channels = num_channels, pool_size = 1024)

    # initialize model
    model = CeA_BASE_1CNN(device = device, grow_on_k_iter=GROW_ON_K_ITER, background_intensity=BACKGROUND_INTENSITY, 
    step_size=STEP_SIZE, scale_mask=SCALE_MASK, pretrain_thres=PRETRAIN, ch0_1=CH0_1, ch1_16=CH1_16, alive_thresh=ALIVE_THRESH)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1500,2500], gamma=0.1) ## keep 1e-4 longer
    model_str = correct_label_in_plot(model)
    target = torch.tensor(target.transpose(-1,0,1)).unsqueeze(0).to(device)
    target_batch = torch.repeat_interleave(target, repeats = sample_size, dim = 0)

    losses = []
    alive_masks = []
    others=[]

    # train automata
    start = time()
    inner_iter_aux = 0
    inner_iter = ITER_INNER
    inner_iters=[]

    for i in range(epochs):
        if i%100 ==0: print(f'epoch={i}')
        if ITER_VAR == 0: # 0 for increasing, 1 for constant
            inner_iter, inner_iter_aux = epochs_in_inner_loop(i, inner_iter_aux, inner_iter)
        inner_iters.append(inner_iter)

        scheduler.step()
        batch_idx = np.random.choice(len(seed_pool), sample_size, replace = False)
        seed_batch = seed_pool[batch_idx].to(device)
        seed_batch[:1] = seed_tensor.to(device)
        
        loss, out, alive_mask, other = model.train_step(
            seed = seed_batch,
            target = target_batch, 
            target_loss_func = F.mse_loss, 
            epochs_inside = inner_iter,
            epoch_outside = i,
            masked_loss = False
            )
        
        alive_masks.append(alive_mask)
        others.append(other)

        seed_pool[batch_idx] = out.detach().to(device)
        loss.backward() # calculate gradients
        model.normalize_grads() # normalize them
        optimizer.step() # update weights and biases 
        optimizer.zero_grad() # prevent accumulation of gradients
        losses.append(loss.item())
        #early-stopping
        if loss.item() < 1e-5: break

        if i % 50==0 or i  == epochs-1:
            model_str_final = plot_loss_and_lesion_synthesis(losses, optimizer, model_str, i, loss, sample_size, out, no_plot=True)

    losses_per_lesion.append(losses)
    stop = time(); time_total = f'{(stop-start)/60:.1f} mins'; print(time_total)
    model_str_final = model_str_final + f'\nep={i}, {time_total}' # for reconstruction figure
    #save model
    torch.save(model.model.state_dict(), f'{path_save_synthesis}{name_prefix}_weights_{idx_lesion:02d}.pt')

    #lesion synthesis
    x = torch.tensor(seed).permute(0,-1,1,2).to(device)
    outs = []
    with torch.no_grad():
        for i,special_sequence in zip(range(256),[1,1,1,3]*64):
            # x = model(x,special_sequence,101)
            x, alive_mask_, others_ = model(x,i,101)
            out = np.clip(to_rgb(x[0].permute(-2, -1,0).cpu().detach().numpy()), 0,1)
            outs.append(out)

    #save results    
    outs_masked = []
    for out_ in outs:
        out_masked = np.squeeze(out_) * target[0,1,...].detach().cpu().numpy()
        out_masked[out_masked==1]=0
        outs_masked.append(out_masked)
    outs_float = np.asarray(outs_masked)
    # print(np.shape(outs_float))
    outs_float = np.clip(outs_float, 0 ,1)
    # outs_int = (outs_int*255).astype('int16')
    # print(f'idx_lesion done = {idx_lesion}')
    
    np.savez_compressed(f'{path_save_synthesis}{name_prefix}_lesion_{idx_lesion:02d}.npz', outs_float)
    np.save(f'{path_save_synthesis}{name_prefix}_coords_{idx_lesion:02d}.npy', coord)
    np.savez_compressed(f'{path_save_synthesis}{name_prefix}_mask_{idx_lesion:02d}.npz', mask)
    np.save(f'{path_save_synthesis}{name_prefix}_loss_{idx_lesion:02d}.npy', losses)
    np.save(f'{path_save_synthesis}{name_prefix}_time_{idx_lesion:02d}_{time_total}.npy', time_total)

# %%
plt.semilogy(losses)
# %%
a0 = np.load(f'{path_save_synthesis}{name_prefix}_loss_00.npy')
a1 = np.load(f'{path_save_synthesis}{name_prefix}_loss_01.npy')
a2 = np.load(f'{path_save_synthesis}{name_prefix}_loss_02.npy')
a3 = np.load(f'{path_save_synthesis}{name_prefix}_loss_03.npy')
# %%
for i in losses_per_lesion:
    plt.semilogy(i)
# %%
np.shape(target_batch)
# %%
