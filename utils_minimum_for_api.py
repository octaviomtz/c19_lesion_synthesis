import math
import numpy as np
import os
from skimage.morphology import remove_small_holes, remove_small_objects
from PIL import Image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion
import streamlit as st
import base64

def original_slice_to_cannonical_view(img, normalize_scan_hounsfield = True, to_0_255=True, rotate=-1, fliplr=True):
    """
    Transform from hounsfield units to a normalized (0-255)
    image that streamlit can plot
    Args:
        

    Returns:
        img [int]: Image ready to plot
    """
    if normalize_scan_hounsfield:
        img = normalize_scan(img)
    if to_0_255:
        img = img*255
    if rotate:
        img = np.rot90(img, rotate)
    if fliplr:
        img = np.fliplr(img)
    return img

def normalize_scan(image, MIN_BOUND=-1000, MAX_BOUND=500):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def from_scan_to_3channel2(img):
    img = np.expand_dims(img,-1)
    img = np.repeat(img,3,-1)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img

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
    return mask_slic, boundaries, segments

def superpixels(im2, segments, background_threshold=.2, vessel_threshold=.4):
  '''1) segment all image using superpixels. 
  2) Then, classify each superpixel into background, vessel or lession according
  to its median intensity'''
  background = np.zeros_like(im2)
  vessels = np.zeros_like(im2)
  lesion_area = np.zeros_like(im2)
  label_background, label_vessels, label_lession = 1, 1, 1,
  for (i, segVal) in enumerate(np.unique(segments)):
    mask = np.zeros_like(im2)
    mask[segments == segVal] = 1
    clus = im2*mask
    median_intensity = np.median(clus[clus>0])
    yy,xx = np.where(mask==1)
    if median_intensity < background_threshold or math.isnan(median_intensity):
      background[yy,xx] = label_background
      label_background += 1
    elif median_intensity > vessel_threshold:
      vessels[yy,xx] = label_vessels
      label_vessels += 1
    else:
      lesion_area[yy,xx] = label_lession
      label_lession += 1
  return background, lesion_area, vessels

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
    plt.axis('off')
    fig_zoom3 = plt.figure()
    plt.imshow(boundaries, cmap='gray')
    plt.axis('off')
    return fig_zoom1, fig_zoom2, fig_zoom3

# @st.cache(suppress_st_warning=True)
def load_synthetic_texture(path_synthesis = '/content/drive/My Drive/Datasets/covid19/results/cea_synthesis/patient0/'):
    texture_orig = np.load(f'{path_synthesis}texture.npy.npz')
    texture_orig = texture_orig.f.arr_0
    texture = texture_orig + np.abs(np.min(texture_orig))# + .07
    return texture

def open_local_gif(location='images/for_gifs/synthesis.gif'):
    """open gif from local file
    from https://discuss.streamlit.io/t/how-to-show-local-gif-image/3408/9"""
    file_ = open(location, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return data_url

# def replace_with_nCA(scan, SCAN_NAME, SLICE, texture, mask_outer_ring = False, POST_PROCESS = True, blur_lesion = False, TRESH_PLOT=10):
#     scan_slice = scan
#     path_parent = '/content/drive/My Drive/Datasets/covid19/COVID-19-20_augs_cea/'
#     path_synthesis_ = f'{path_parent}CeA_BASE_grow=1_bg=-1.00_step=-1.0_scale=-1.0_seed=1.0_ch0_1=-1_ch1_16=-1_ali_thr=0.1/'
#     path_synthesis = f'{path_synthesis_}{SCAN_NAME}/'
#     # path_synthesis = '/content/drive/My Drive/repositories/cellular_automata/Growing-Neural-Cellular-Automata/temp_delete/volume-covid19-A-0014/'
#     lesions_all, coords_all, masks_all, names_all, loss_all = read_cea_aug_slice2(path_synthesis, SLICE=SLICE)
#     V_MAX = np.max(scan_slice)
#     slice_healthy_inpain = pseudo_healthy_with_texture(scan_slice, lesions_all, coords_all, masks_all, names_all, texture)
#     decreasing_sequence = get_decreasing_sequence(255, splits= 20) 
#     # decreasing_sequence = np.arange(0,30)
#     arrays_sequence = []
#     images=[]
#     mse_gen = []
#     st.write(np.max(scan), np.max(texture), np.max(lesions_all[0]))
#     for GEN in decreasing_sequence:

#         slice_healthy_inpain2 = copy(slice_healthy_inpain)
#         synthetic_intensities=[]
#         mask_for_inpain = np.zeros_like(slice_healthy_inpain2)
#         mse_lesions = []
#         for idx_x, (lesion, coord, mask, name) in enumerate(zip(lesions_all, coords_all, masks_all, names_all)):
#             #get the right coordinates
#             coords_big2 = [int(i) for i in name.split('_')[1:5]]
#             coords_sums = coord + coords_big2
#             new_coords_mask = np.where(mask==1)[0]+coords_sums[0], np.where(mask==1)[1]+coords_sums[2]
#             if GEN<60:
#                 if POST_PROCESS:
#                     syn_norm = normalize_new_range4(lesion[GEN], scan_slice[new_coords_mask], scale=.5)
#                 else:
#                     syn_norm = lesion[GEN]
#             else:
#                 syn_norm = lesion[GEN]
#             # get the MSE between synthetic and original
#             orig_lesion = get_orig_scan_in_lesion_coords(scan_slice, new_coords_mask)
#             mse_lesions.append(np.mean(mask*(syn_norm - orig_lesion)**2))

#             syn_norm = syn_norm * mask
#             if blur_lesion:
#                 syn_norm = blur_masked_image(syn_norm, kernel_blur=(2,2))
#             # add cea syn with absolute coords
#             new_coords = np.where(syn_norm>0)[0]+coords_sums[0], np.where(syn_norm>0)[1]+coords_sums[2]
#             slice_healthy_inpain2[new_coords] = syn_norm[syn_norm>0]

#             synthetic_intensities.extend(syn_norm[syn_norm>0])

#             # inpaint the outer ring
#             if mask_outer_ring:
#                 mask_ring = make_mask_ring(syn_norm>0)
#                 new_coords_mask_inpain = np.where(mask_ring==1)[0]+coords_sums[0], np.where(mask_ring==1)[1]+coords_sums[2] # mask outer rings for inpaint
#                 mask_for_inpain[new_coords_mask_inpain] = 1
        
#         mse_gen.append(mse_lesions)
#         if mask_outer_ring:
#             slice_healthy_inpain2 = inpaint.inpaint_biharmonic(slice_healthy_inpain2, mask_for_inpain)

#         # arrays_sequence.append(slice_healthy_inpain2[coords_big2[0]-TRESH_PLOT:coords_big2[1]+TRESH_PLOT,coords_big2[2]-TRESH_PLOT:coords_big2[3]+TRESH_PLOT])
#         arrays_sequence.append(slice_healthy_inpain2)
#         # images.append(slice_healthy_inpain2[coords_big2[0]-TRESH_PLOT:coords_big2[1]+TRESH_PLOT,coords_big2[2]-TRESH_PLOT:coords_big2[3]+TRESH_PLOT])
#         # Create GIF
#         path_gifs = "images/for_gifs/"
#         scan_name_with_slice = f'{SCAN_NAME}_{SLICE}.gif'
#         if scan_name_with_slice not in os.listdir(path_gifs):
#             _ = fig_blend_lesion(slice_healthy_inpain2, coords_big2, GEN, decreasing_sequence, path_synthesis, len(lesion), file_path=f"{path_gifs}synthesis.png", Tp=30, V_MAX=V_MAX)
#             _ = images.append(imageio.imread(f"{path_gifs}synthesis.png")); #Adds images to list
#     if scan_name_with_slice not in os.listdir(path_gifs):
#         imageio.mimsave(f"{path_gifs}{SCAN_NAME}_{SLICE}.gif", images, fps=4) #Creates gif out of list of images
#         print('DOING GIF')
#     else:
#         print('GIF ALREADY DONE')
#     # _ = plt.close()

#     return arrays_sequence, decreasing_sequence

def coords_min_max_2D(array):
  '''return the min and max+1 of a mask. We use mask+1 to include the whole mask'''
  yy, xx = np.where(array==True)
  y_max = np.max(yy)+1; y_min = np.min(yy)
  x_max = np.max(xx)+1; x_min = np.min(xx)
  return y_min, y_max, x_min, x_max