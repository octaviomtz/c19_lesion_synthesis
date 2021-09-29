#%%
import numpy as np
import matplotlib.pyplot as plt
from utils_load import (
    load_synthetic_lesions_scans_and__individual_lesions,
    superpixels_applied,
    load_only_original_scans,
    normalize_scan,
    from_scan_to_3channel,
    load_synthetic_texture,
    original_slice_to_cannonical_view,
)
from utils import coords_min_max_2D
from scipy.spatial import distance
from scipy.ndimage.morphology import binary_erosion, binary_dilation, distance_transform_bf
from scipy.ndimage import label
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import torch
from copy import copy
from scipy.ndimage import distance_transform_bf
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torchvision.models as models
import torch
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tnrange
import moviepy.editor as mvp

#%%
def load_initially(SCAN_NAME):
    # scan_3D = np.load(f'images/scans/{SCAN_NAME}_old.npy')
    scan_3D = np.load(f'images/scans/{SCAN_NAME}.npz')
    scan_3D = scan_3D.f.arr_0
    scan_mask_3D = np.load(f'images/masks/{SCAN_NAME}.npz')
    scan_mask_3D = scan_mask_3D.f.arr_0
    slices_with_lesions = np.where(np.sum(scan_mask_3D, (0,1))>20)[0] 
    # st.write(slices_with_lesions, np.min(st.session_state['scan']),np.max(st.session_state['scan']))
    
    # fig = plt.figure()
    # plt.imshow(st.session_state['scan'])
    # st.pyplot(fig)
    texture = load_synthetic_texture()
    return scan_3D, scan_mask_3D, texture

def load_slice_initially(scan, scan_mask, ONLY_ONE_SLICE=34):
    scan_slice = scan[...,ONLY_ONE_SLICE]
    scan_mask = scan_mask[...,ONLY_ONE_SLICE]
    return scan_slice, scan_mask

def find_closest_cluster(boundaries, label_):
    '''Find the number of the cluster that is closer to 
    any point of cluster label_
    returns:
    -the label of the closest cluster
    -the closest coord of the closest cluster
    -the shortest distance'''
    XX = np.where(boundaries==label_)
    cluster0_coords = np.asarray([np.asarray((i,j)) for i,j in zip(XX[0], XX[1])])
    XX = np.where(np.logical_and(boundaries!=label_, boundaries>0))
    cluster_others_coords = np.asarray([np.asarray((i,j)) for i,j in zip(XX[0], XX[1])])
    dists = distance.cdist(cluster0_coords, cluster_others_coords)#.min(axis=1)
    min_dist = np.min(dists.min(axis=0))
    dist_small = np.where(dists==min_dist)[1][0]
    closest_coord = cluster_others_coords[dist_small]
    closest_cluster = boundaries[closest_coord[0],closest_coord[1]]
    return closest_cluster, closest_coord, min_dist

def find_texture_relief(area, thresh0=.3, thresh1=.15):
    area_mod = binary_erosion(area > thresh0)
    area_mod1 = binary_dilation(area_mod)
    area_mod1 = distance_transform_bf(area_mod)
    area_mod = binary_erosion(area > thresh1)
    area_mod = binary_dilation(area_mod)
    area_mod = distance_transform_bf(area_mod)
    xx = area_mod+area_mod1*2
    labelled, nr = label(xx)
    xx = labelled * ((area_mod>0).astype(int)-(binary_erosion(area_mod>0)).astype(int))
    return xx, labelled, area_mod

def get_cluster_borders(segments, cluster_sorted, clusters_total=5):
    cluster_borders = np.zeros_like(segments).astype(float)
    for i in range(clusters_total): 
        cluster_number = cluster_sorted[i][0]
        cluster_i = segments == cluster_number
        cluster_i_border = cluster_i - binary_erosion(cluster_i).astype(float)
        cluster_i = cluster_i * cluster_number
        cluster_i_border = cluster_i_border * cluster_number
        cluster_borders+= cluster_i_border
    return cluster_borders

class VideoWriter:
  def __init__(self, filename='_autoplay.mp4', fps=20.0, **kw): #XX fps=30.0
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()
    if self.params['filename'] == '_autoplay.mp4':
      self.show()

  def show(self, **kw):
      self.close()
      fn = self.params['filename']
      display(mvp.ipython_display(fn, **kw))

# %%
scan_selection = 14
SCAN_NAME = f'volume-covid19-A-{scan_selection:04d}'
scan_3D, scan_mask_3D, texture = load_initially(SCAN_NAME)
scan_slice, scan_mask = load_slice_initially(scan_3D, scan_mask_3D)
coords = coords_min_max_2D(scan_mask)

#%%
scan_norm = original_slice_to_cannonical_view(scan_slice, to_0_255=False)
scan_lesion = (scan_norm*scan_mask)[coords[0]:coords[1],coords[2]:coords[3]]
plt.imshow(scan_lesion)
np.min(scan_lesion), np.max(scan_lesion)

#%% SUPERPIXELS
numSegments = 40
segments = slic((scan_lesion).astype('double'), 
                n_segments = numSegments, 
                mask=scan_mask[coords[0]:coords[1],coords[2]:coords[3]], 
                sigma = .2, multichannel=False, compactness=.1)
boundaries = mark_boundaries(scan_lesion, segments)[...,0]
fig, ax = plt.subplots(2,2, figsize=(8,8))
ax[0,0].imshow(scan_lesion)
ax[0,1].imshow(scan_lesion)
ax[0,1].imshow(boundaries)
ax[1,0].imshow(segments)
ax[1,1].imshow(segments)

#%% SORT SUPERPIXELS BY MEDIAN INTENSITY
cluster_sorted = []
for idx, i in enumerate(np.unique(segments)):
    if idx ==0: continue
    cluster_i = scan_lesion * (segments==i)
    median_value = np.median(cluster_i[np.where(cluster_i>0)])
    cluster_sorted.append(np.asarray([i, median_value]))
cluster_sorted = np.asarray(cluster_sorted)
# sort array
cluster_sorted = cluster_sorted[cluster_sorted[:, 1].argsort()[::-1]]

#=== JOIN CLOSE SUPERPIXELS AND APPLY DISTANCE TRANSFORM
segments2 = copy(segments)
cluster_sorted2 = cluster_sorted
len(cluster_sorted2)

superpix_join=[]
superpix_added=[]
mask_df0 = distance_transform_bf(segments2==cluster_sorted2[0,0])
mask_df0 = 1/(1+(np.exp(-mask_df0*1)))
mask_df0 = (mask_df0-np.min(mask_df0))/(np.max(mask_df0)-np.min(mask_df0))
superpix_added.append(mask_df0.astype(float))
check_n_clus = 5
unique_len_segments = len(np.unique(segments))
for i in range(len(cluster_sorted2)-2):
    check_n_clus2 = [check_n_clus if len(cluster_sorted2) > check_n_clus else len(cluster_sorted2)-1][0]
    cluster_borders = get_cluster_borders(segments2, cluster_sorted2, check_n_clus2)
    cluster_closer, cluster_closer_coord, cluster_closer_dist =  find_closest_cluster(cluster_borders, cluster_sorted2[0,0])
    segments2[np.where(segments2==cluster_closer)] = cluster_sorted2[0,0]
    row_delete = np.where(cluster_sorted2[:,0]==cluster_closer)[0][0]
    cluster_sorted2 = np.delete(cluster_sorted2, (row_delete), axis=0)
    mask_df = distance_transform_bf(segments2==cluster_sorted2[0,0])
    mask_df = 1/(1+(np.exp(-mask_df*1)))
    mask_df = (mask_df-np.min(mask_df))/(np.max(mask_df)-np.min(mask_df))
    superpix_join.append(segments2/unique_len_segments)
    superpix_added.append((mask_df).astype(float))
mask_df_last = distance_transform_bf(segments2>0)
mask_df_last = 1/(1+(np.exp(-mask_df_last*1)))
mask_df_last = (mask_df_last-np.min(mask_df_last))/(np.max(mask_df_last)-np.min(mask_df_last))
superpix_added.append(mask_df_last.astype(float))

#%%
fig, ax = plt.subplots(8,5, figsize=(12,16))
for i in range(len(superpix_added)):
  ax.flat[i].imshow(superpix_added[i])
  # ax.flat[i].hist(superpix_added[i].flatten())
  # ax.flat[i].set_xlim([0,1])

#%% NCA =========
ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
lap = torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])

def perchannel_conv(x, filters):
  '''filters: [filter_n, h, w]'''
  b, ch, h, w = x.shape
  y = x.reshape(b*ch, 1, h, w)
  y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
  y = torch.nn.functional.conv2d(y, filters[:,None])
  return y.reshape(b, -1, h, w)

def perception(x):
  filters = torch.stack([ident, sobel_x, sobel_x.T, lap])
  return perchannel_conv(x, filters)

class CA(torch.nn.Module):
  def __init__(self, chn=12, hidden_n=96):
    super().__init__()
    self.chn = chn
    self.w1 = torch.nn.Conv2d(chn*4, hidden_n, 1)
    self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
    self.w2.weight.data.zero_()

  def forward(self, x, update_rate=0.5, noise=None):
    if noise is not None:
      x += torch.randn_like(x)*noise
    y = perception(x)
    y = self.w2(torch.relu(self.w1(y)))
    b, c, h, w = y.shape
    udpate_mask = (torch.rand(b, 1, h, w)+update_rate).floor()
    return x+y*udpate_mask

  def seed(self, n, sz=128):
    return torch.zeros(n, self.chn, sz, sz)

def to_rgb(x):
  return x[...,:3,:,:]+0.5

#%%
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# ca = torch.load('models/ca_lungs_covid2.pt')
ca1 = torch.load('models/init_lungs.pt').cpu()
ca2 = torch.load('models/init_lungs_covid_higher_2.5.pt').cpu()


#%% MASKS REPEATED ACROSS ALL SYNTHESIS
len_masks = np.shape(superpix_added)[0]
len_synthesis = 400
masks_repeated_ = []
idxs_ = []
for i in range(len_masks):
    for j in range((len_synthesis//(len_masks*2))):
        masks_repeated_.append(superpix_added[i])
        idxs_.append(i)
masks_repeated = masks_repeated_ + masks_repeated_[::-1]
idxs = idxs_ + idxs_[::-1]
# for i in range(len_masks):
#     if i==0: continue
#     for j in range((len_synthesis//(len_masks*2))):
#         masks_repeated.append(superpix_added[-i])
#         idxs.append(-i)
# for j in range((len_synthesis//(len_masks*2))):
#     masks_repeated.append(np.zeros_like(superpix_added[0]))
#     idxs.append(-i)
plt.plot(idxs)

#%% GRAFTING
H, W = 80, 80 
hh, ww = np.shape(superpix_added[30])
imgs = []
masks_list = []
# with VideoWriter('graft.mp4') as vid, torch.no_grad():
with torch.no_grad():
  x = torch.zeros([1, ca1.chn, H, W])
  for i in tnrange(400):
    mask = torch.zeros((H, W)).clone().detach()
    ii = [i%100 if i%100 < len(superpix_added) else 0][0]
    mask[20:20+hh, 20:20+ww] = torch.tensor(masks_repeated[i]).clone().detach()
    masks_list.append(mask)
    for k in range(8): #48
      x1, x2 = ca1(x), ca2(x)
      x = x1 + (x2-x1)*mask
      # x = x1*(1-mask) + x2*(mask)
    img = to_rgb(x[0]).permute(1, 2, 0)[...,0].clone().numpy()#detach().cpu().numpy()
    # vid.add(zoom(img, 2))
    imgs.append(img)

np.shape(imgs)

#%% ================ FIGURES
fig, ax = plt.subplots(6,12, figsize=(24,12))
for i in range(72):
    ii = i+194
    ax.flat[i].imshow(imgs[ii], vmin=0, vmax=1)
    # ax.flat[i].imshow(masks_list[ii].cpu().numpy(), alpha=.3)
    ax.flat[i].text(10, 10, ii, c='r', fontsize=12)
    ax.flat[i].axis('off')
ax.flat[i].imshow(masks_list[ii].cpu().numpy())
fig.tight_layout()

#%%
scan_norm_patch = copy(scan_norm)
mm = masks_list[200][20:20+hh, 20:20+ww]
mm_y, mm_x = np.where(mm>0)
scan_norm_patch[mm_y+coords[0], mm_x+coords[2]] = imgs[200][mm_y+20, mm_x+20]
sp=10
# scan_norm_patch[coords[0]:coords[1],coords[2]:coords[3]] = img[20:20+hh, 20:20+ww]
fig, ax = plt.subplots(2,3, figsize=(12,8))
ax[0,0].imshow(scan_norm)
ax[0,1].imshow(scan_mask)
ax[1,0].imshow(scan_norm[coords[0]:coords[1],coords[2]:coords[3]])
ax[1,1].imshow(scan_mask[coords[0]:coords[1],coords[2]:coords[3]])
ax[0,2].imshow(scan_norm_patch)
ax[1,2].imshow(scan_norm_patch[coords[0]-sp:coords[1]+sp,coords[2]-sp:coords[3]+sp])

#%%
scan_norm_patch = copy(scan_norm)
mm = masks_list[200][20:20+hh, 20:20+ww]
mm_y, mm_x = np.where(mm>0)



plt.imshow(scan_norm_patch)


#%%
fig, ax = plt.subplots(6,12, figsize=(24,12))
for i in range(72):
    ii= i+4 #i*5
    ax.flat[i].imshow(masks_repeated[ii])
    ax.flat[i].text(10, 10, ii, c='r', fontsize=12)
    ax.flat[i].axis('off')


#%%
img1 = to_rgb(x1[0]).permute(1, 2, 0)[...,0].cpu().numpy()
img2 = to_rgb(x2[0]).permute(1, 2, 0)[...,0].cpu().numpy()
img3 = (img2-img1)* mask.cpu().numpy()
img4 = img1 + img3
fig, ax = plt.subplots(2,3, figsize=(12,6))
ax[0,0].imshow(img1, vmin=0, vmax=1)
ax[0,1].imshow(img2, vmin=0, vmax=1)
ax[0,2].imshow(mask_changing.cpu().numpy())
ax[1,0].imshow(img3)
ax[1,1].imshow(img4, vmin=0, vmax=1)
ax[1,2].imshow(img, vmin=0, vmax=1)
np.min(img3), np.max(img3)
plt.imshow(masks_list[320].cpu())
328%100
#%%
plt.imshow(mask_changing.cpu().numpy())
plt.colorbar()

#%%
fig, ax = plt.subplots(6,6, figsize=(12,12))
for i in range(36):
  ii = [i%100 if i%100 < len(superpix_added) else 0][0]
  ax.flat[i].imshow(masks_list[300+i])


#%%








#%% NOISE-CONTROLLED TEXTURE
h, w = 100, 100 # np.shape(noise_frames[-4])#
noise = 0.5-0.5*torch.linspace(0, np.pi*2.0, w).cos()
noise *= 0.02
noise_temp = np.zeros((h,w))
all_noise = np.ones((h,w))*.02
hh, ww = np.shape(superpix_added[30])
k_counter = 0
imgs = []
masks = []
ks = []
# noise_temp[100:100+hh, 100:100+ww] = noise_frames[-4]*.02
with VideoWriter('covid_x.mp4') as vid, torch.no_grad():
  x = torch.zeros(1, ca.chn, h, w)
  for k in tnrange(50, leave=False):
    # noise_temp[40:40+hh, 30:30+ww] = superpix_added[30]*.02 #noise_dt*.02 
    # x[:] = ca(x, noise=torch.tensor(noise_temp))
    x[:] = ca(x)
  for k in tnrange(250, leave=False):
    if k<50:# or k>100:
      for kk in range(3):
        x[:] = ca(x)
    else:
      if k % (250//len(superpix_added)) == 0: 
        k_counter+=1
        print(k, k_counter, np.sum(superpix_added[k_counter]))
      ks.append(k_counter)
      noise_temp[40:40+hh, 30:30+ww] = superpix_added[k_counter]*.02 #noise_dt*.02 
      masks.append(noise_temp)
      for kk in range(10):
        x[:] = ca(x, noise=torch.tensor(noise_temp))
        
    img = to_rgb(x[0]).permute(1, 2, 0)[...,0].cpu() 
    vid.add(img) 
    imgs.append(img)
    
vid.show(loop=True)

#%%
np.sum(superpix_added[1])
plt.hist(superpix_added[1].flatten())

#%% PLOT LAST IMAGE
print(np.unique(noise_temp))
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].imshow(img)
ax[1].imshow(noise_temp)
np.max(img.numpy())
len(imgs), len(masks)
plt.figure()
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].imshow(masks[1])
ax[1].imshow(superpix_added[3])
k, k_counter

# %% PLOT SEVERAL SYNTHETIC IMAGES
fig, ax = plt.subplots(8,5, figsize=(12,20))
for i in range(40):
    # ax.flat[i].imshow(imgs[i*8], vmin=0, vmax=1);
    ax.flat[i].imshow(masks[i]);
    ax.flat[i].axis('off');
fig.tight_layout()



#%% 
cluster_borders = np.zeros_like(scan_lesion)
clusters_total=5
for i in range(clusters_total): 
    cluster_number = cluster_sorted[i][0]
    cluster_i = segments == cluster_number
    cluster_i_border = cluster_i - binary_erosion(cluster_i).astype(int)
    cluster_i = cluster_i * cluster_number
    cluster_i_border = cluster_i_border * cluster_number
    cluster_borders+= cluster_i_border
plt.imshow(cluster_borders)

#%%
cluster_closer, cluster_closer_coord, cluster_closer_dist =  find_closest_cluster(cluster_borders, cluster_sorted[0,0])
cluster_closer

#%%
print(np.unique(cluster_borders))
print(cluster_sorted[:10,0])
cluster_closer, cluster_closer_coord, cluster_closer_dist =  find_closest_cluster(cluster_borders, cluster_sorted[0,0])
cluster_closer


#%%
from copy import copy
segments2 = copy(segments)
cluster_sorted2 = cluster_sorted
plt.imshow(segments2)
cluster_sorted2[:5]
#%%
with VideoWriter('join_superpixels.mp4') as vid:
    for i in range(35):
        cluster_borders = get_cluster_borders(segments2, cluster_sorted2)
        cluster_closer, cluster_closer_coord, cluster_closer_dist =  find_closest_cluster(cluster_borders, cluster_sorted2[0,0])
        segments2[np.where(segments2==cluster_closer)] = cluster_sorted2[0,0]
        row_delete = np.where(cluster_sorted2[:,0]==cluster_closer)[0][0]
        cluster_sorted2 = np.delete(cluster_sorted2, (row_delete), axis=0)
        vid.add(segments2/40)
        # plt.figure()
        # plt.imshow(segments2)

#%%
print((cluster_sorted[0,0]))
segments2[np.where(segments2==cluster_closer)] = cluster_sorted[0,0]
plt.imshow(segments2)

#%%
cluster_sorted2 = cluster_sorted
print(cluster_sorted2[:5])
row_delete = np.where(cluster_sorted[:,0]==cluster_closer)[0][0]
cluster_sorted2 = np.delete(cluster_sorted2, (row_delete), axis=0)
print(cluster_sorted2[:5])

#%% ============================
#==============

# %%
print(img.shape, segments2.shape)
print(np.max(img.numpy()), np.max(segments2))
# %%
scan2 = np.swapaxes(scan,0,1)
scan2 = normalize_scan(scan2)
scan2 = (scan2*255).astype(np.uint8)
plt.imshow(scan2[...,34])
# %%
np.save('images/scans/volume-covid19-A-0014.npy', scan2)
# %%
scan_mask2 = np.swapaxes(scan_mask,0,1)
np.savez_compressed('images/masks/volume-covid19-A-0014',scan_mask2)
# %%




#%% WORKING PARAMETERS
h, w = 1080//4, 1920//4 # np.shape(noise_frames[-4])#
noise = 0.5-0.5*torch.linspace(0, np.pi*2.0, w).cos()
noise *= 0.02
noise_temp = np.zeros((h,w))
all_noise = np.ones((h,w))*.02
hh, ww = np.shape(noise_frames[-4])
# noise_temp[100:100+hh, 100:100+ww] = noise_frames[-4]*.02
with VideoWriter('covid_x.mp4') as vid, torch.no_grad():
  x = torch.zeros(1, ca.chn, h, w)
  for k in tnrange(50, leave=False):
    noise_temp[100:100+hh, 100:100+ww] = noise_dt*.02 #noise_frames_repeated1[160]
    x[:] = ca(x, noise=torch.tensor(noise_temp))
    # x[:] = ca(x)
  for k in tnrange(200, leave=False):
    for kk in range(5):
      if k<10:
        x[:] = ca(x)
      else:
        noise_temp[100:100+hh, 100:100+ww] = noise_dt*.02 #noise_frames_repeated1[260]
        x[:] = ca(x, noise=torch.tensor(noise_temp))
    img = to_rgb(x[0]).permute(1, 2, 0)[...,0].cpu() # these 2 row were idented left
    vid.add(img) # these 2 row were idented left
vid.show(loop=True)

#%%
def find_closest_cluster(boundaries, label_):
    '''Find the number of the cluster that is closer to 
    any point of cluster label_
    returns:
    -the label of the closest cluster
    -the closest coord of the closest cluster
    -the shortest distance'''
    XX = np.where(boundaries==label_)
    cluster0_coords = np.asarray([np.asarray((i,j)) for i,j in zip(XX[0], XX[1])])
    XX = np.where(np.logical_and(boundaries!=label_, boundaries>0))
    cluster_others_coords = np.asarray([np.asarray((i,j)) for i,j in zip(XX[0], XX[1])])
    dists = distance.cdist(cluster0_coords, cluster_others_coords)#.min(axis=1)
    min_dist = np.min(dists.min(axis=0))
    dist_small = np.where(dists==min_dist)[1][0]
    closest_coord = cluster_others_coords[dist_small]
    closest_cluster = boundaries[closest_coord[0],closest_coord[1]]
    return closest_cluster, closest_coord, min_dist

def find_texture_relief(area, thresh0=.3, thresh1=.15):
    area_mod = binary_erosion(area > thresh0)
    area_mod1 = binary_dilation(area_mod)
    area_mod1 = distance_transform_bf(area_mod)
    area_mod = binary_erosion(area > thresh1)
    area_mod = binary_dilation(area_mod)
    area_mod = distance_transform_bf(area_mod)
    xx = area_mod+area_mod1*2
    labelled, nr = scipy.ndimage.label(xx)
    xx = labelled * ((area_mod>0).astype(int)-(binary_erosion(area_mod>0)).astype(int))
    return xx, labelled, area_mod

#%%
def isPath(matrix, n):
    '''https://www.geeksforgeeks.org/find-whether-path-two-cells-matrix/'''
    # Defining visited array to keep
    # track of already visited indexes
    visited = [[False for x in range (n)]
                      for y in range (n)]
    
    # Flag to indicate whether the
    # path exists or not
    flag = False
 
    for i in range (n):
        for j in range (n):
           
            # If matrix[i][j] is source
            # and it is not visited
            if (matrix[i][j] == 1 and not
                visited[i][j]):
 
                # Starting from i, j and
                # then finding the path
                if (checkPath(matrix, i,
                              j, visited)):
                   
                    # If path exists
                    flag = True
                    break
    if (flag):
        print("YES")
    else:
        print("NO")
 
# Method for checking boundaries
def isSafe(i, j, matrix):
   
    if (i >= 0 and i < len(matrix) and
        j >= 0 and j < len(matrix[0])):
        return True
    return False
 
# Returns true if there is a
# path from a source(a
# cell with value 1) to a
# destination(a cell with
# value 2)
def checkPath(matrix, i, j,
              visited):
 
    # Checking the boundaries, walls and
    # whether the cell is unvisited
    if (isSafe(i, j, matrix) and
        matrix[i][j] != 0 and not
        visited[i][j]):
       
        # Make the cell visited
        visited[i][j] = True
 
        # If the cell is the required
        # destination then return true
        if (matrix[i][j] == 2):
           return True
 
        # traverse up
        up = checkPath(matrix, i - 1,
                       j, visited)
 
        # If path is found in up
        # direction return true
        if (up):
           return True
 
        # Traverse left
        left = checkPath(matrix, i,
                         j - 1, visited)
 
        # If path is found in left
        # direction return true
        if (left):
           return True
 
        # Traverse down
        down = checkPath(matrix, i + 1,
                         j, visited)
 
        # If path is found in down
        # direction return true
        if (down):
           return True
 
        # Traverse right
        right = checkPath(matrix, i,
                          j + 1, visited)
 
        # If path is found in right
        # direction return true
        if (right):
           return True
     
    # No path has been found
    return False

# %%

print(np.unique(superpix_added[0]))
# plt.hist(superpix_added[0].flatten()); #XX CONTINUE HERE

#%%
# labelled_boundaries, labelled, area_mod = find_texture_relief(scan_lesion)
# fig, ax = plt.subplots(1,3, figsize=(12,8))
# ax[0].imshow(scan_lesion)
# ax[1].imshow(area_mod)
# ax[2].imshow(labelled_boundaries)

# # %%
# print(coords)
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(scan_norm)
# ax[1].imshow(scan_norm[coords[0]:coords[1],coords[2]:coords[3]])
# ax[1].imshow(scan_mask[coords[0]:coords[1],coords[2]:coords[3]], alpha=.3)

# #%%
# bin_info = plt.hist(scan_lesion[scan_lesion>0], bins=20)
# plt.close()
# bin_count, bin_lim = bin_info[0], bin_info[1]
# bin_lim = np.insert(bin_lim,0,0)
# noise_frames = []
# for i in bin_lim[::-1]:
#     noise_frames.append((scan_lesion>=i) * 0.02) 
# fig, ax = plt.subplots(4,5,figsize=(12,8))
# for idx in range(20):
#     ax.flat[idx].imshow(noise_frames[idx])
#     ax.flat[idx].axis('off')
#%%
# repetitions_per_frame1 = int(np.ceil(1000/np.shape(noise_frames)[0]))
# repetitions_per_frame2 = int(np.ceil(600/np.shape(noise_frames)[0]))
# repetitions_per_frame1, repetitions_per_frame2
# noise_frames_repeated1 = []
# noise_frames_repeated2 = []
# for n_frame in noise_frames:
#     for it in range(repetitions_per_frame1):
#         noise_frames_repeated1.append(n_frame)
#         if it <= repetitions_per_frame2:
#             noise_frames_repeated2.append(n_frame)


#%%
# np.shape(noise_frames_repeated1), np.shape(noise_frames_repeated2)
# print(len(noise_frames))
# fig, ax = plt.subplots(5,5)
# for i in range(25):
#   ax.flat[i].imshow(noise_frames[i])

# from scipy.ndimage import distance_transform_bf
# import math
# noise_dt = distance_transform_bf(noise_frames[-2])
# noise_dt = noise_dt / np.max(noise_dt)
# print(type(noise_dt))
# noise_dt = (1/(1+np.exp(-noise_dt*10)))
# plt.imshow(noise_dt)
# plt.colorbar()

# #%%
# a=np.linspace(0,1,100)
# b = 1/(1+np.exp(-a))
# c = 1/(1+np.exp(-a*10))
# plt.plot(a,b)
# plt.plot(a,c)