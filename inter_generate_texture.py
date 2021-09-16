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
torch.set_default_tensor_type('torch.cuda.FloatTensor')
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

#%%
def find_closest_cluster(boundaries, label_):
    '''Find the number of the cluster that is closer to 
    any point of cluster label_'''
    XX = np.where(boundaries==label_)
    cluster0_coords = np.asarray([np.asarray((i,j)) for i,j in zip(XX[0], XX[1])])
    XX = np.where(np.logical_and(boundaries!=label_, boundaries>0))
    cluster_others_coords = np.asarray([np.asarray((i,j)) for i,j in zip(XX[0], XX[1])])
    dists = distance.cdist(cluster0_coords, cluster_others_coords)#.min(axis=1)
    dist_small = np.where(dists==np.min(dists.min(axis=0)))[1][0]
    closest_coord = cluster_others_coords[dist_small]
    closest_cluster = boundaries[closest_coord[0],closest_coord[1]]
    return closest_cluster, closest_coord

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

#%%
labelled_boundaries, labelled, area_mod = find_texture_relief(scan_lesion)
fig, ax = plt.subplots(1,3, figsize=(12,8))
ax[0].imshow(scan_lesion)
ax[1].imshow(area_mod)
ax[2].imshow(labelled_boundaries)

# %%
print(coords)
fig, ax = plt.subplots(1,2)
ax[0].imshow(scan_norm)
ax[1].imshow(scan_norm[coords[0]:coords[1],coords[2]:coords[3]])
ax[1].imshow(scan_mask[coords[0]:coords[1],coords[2]:coords[3]], alpha=.3)

#%%
bin_info = plt.hist(scan_lesion[scan_lesion>0], bins=20)
plt.close()
bin_count, bin_lim = bin_info[0], bin_info[1]
bin_lim = np.insert(bin_lim,0,0)
noise_frames = []
for i in bin_lim[::-1]:
    noise_frames.append(scan_lesion>=i)
fig, ax = plt.subplots(4,5,figsize=(12,8))
for idx in range(20):
    ax.flat[idx].imshow(noise_frames[idx])
    ax.flat[idx].axis('off')


#%% NCA =========
import torchvision.models as models
import torch
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tnrange
import moviepy.editor as mvp
#%% NCA TEXTURE
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
class VideoWriter:
  def __init__(self, filename='_autoplay.mp4', fps=10.0, **kw): #XX fps=30.0
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

torch.set_default_tensor_type('torch.cuda.FloatTensor')
#%%
# from PIL import Image
# import PIL
# import torch
# import numpy as np
# h, w = 1080//2, 1920//2
# mask = PIL.Image.new('L', (w, h))
# draw = PIL.ImageDraw.Draw(mask)
# font = PIL.ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf', 300)
# draw  = PIL.ImageDraw.Draw(mask)
# draw.text((20, 100), 'COVID', fill=255, font=font, align='center')
# noise = torch.tensor(np.float32(mask)/255.0*0.02)
# plt.hist(np.asarray(noise.cpu().numpy()).flatten());
#%%
ca = torch.load('models/ca_lungs_covid2.pt')
#%%
h, w = 1080//2, 1920//2 # np.shape(noise_frames[-4])#
noise_temp = np.zeros((h,w))
hh, ww = np.shape(noise_frames[-4])
noise_temp[100:100+hh, 100:100+ww] = noise_frames[-4]*.02
with VideoWriter('covid_x.mp4') as vid, torch.no_grad():
  x = torch.zeros(1, ca.chn, h, w)
  for k in tnrange(1000, leave=False):
    x[:] = ca(x, noise=torch.tensor(noise_temp))
  for k in tnrange(600, leave=False):
    for k in range(10):
      x[:] = ca(x, noise=torch.tensor(noise_temp))
    img = to_rgb(x[0]).permute(1, 2, 0)[...,0].cpu()
    vid.add(img)
vid.show(loop=True)

# %%
img = img.cpu().numpy()
img2 = (img - np.min(img))/ (np.max(img)-np.min(img))
plt.imshow(img2)

#%% SUPERPIXELS
numSegments = 20
segments = slic((scan_lesion).astype('double'), n_segments = numSegments, 
                mask=scan_mask[coords[0]:coords[1],coords[2]:coords[3]], 
                sigma = .2, multichannel=False, compactness=.1)
boundaries = mark_boundaries(scan_lesion, segments)[...,0]
fig, ax = plt.subplots(2,2, figsize=(8,8))
ax[0,0].imshow(scan_lesion)
ax[0,1].imshow(scan_lesion)
ax[0,1].imshow(boundaries)
ax[1,0].imshow(segments)
ax[1,1].imshow(segments)

#%% ============================
#==============

# %%
scan2 = np.swapaxes(scan,0,1)
scan2 = normalize_scan(scan2)
plt.imshow(scan2[...,34])
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




#%%
slc = 34
SCAN_NAME = f'volume-covid19-A-{scan_selection:04d}'
aa = np.load(f'images/scans/{SCAN_NAME}.npy')
aa = aa[...,slc]
np.shape(aa)
# %%
plt.imshow(aa)
# %%
plt.hist(aa.flatten());
# %%
aa2 = np.expand_dims(aa,-1)
aa2 = np.repeat(aa2, 3, -1)
np.shape(aa2)
# %%
plt.imshow(aa2)
#%%
from PIL import Image
aa3 = Image.fromarray(aa2)
# %%
plt.imshow(aa3)
# %%
