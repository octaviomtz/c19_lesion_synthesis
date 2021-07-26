# python3 -m venv venv
# . venv/bin/activate
# pip install streamlit
# pip install torch torchvision
# streamlit run main.py

import streamlit as st
from PIL import Image
from utils_load import (
    load_synthetic_lesions_scans_and__individual_lesions,
    superpixels_applied
)
import numpy as np
import matplotlib.pyplot as plt

def normalize_scan(image, MIN_BOUND=-1000, MAX_BOUND=500):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

st.set_page_config(layout="wide")
# import style
# use https://github.com/andfanilo/streamlit-drawable-canvas

st.title('PyTorch Style Transfer')

a1, a2 = st.beta_columns((1, 1))

option = a2.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'))

c1, c2, c3, c4 = st.beta_columns((2, 1, 1, 1))

path_img_temp = "images/output-images/mosaic-cat.png"
path_img_temp2 = "images/output-images/candy-cat.png"
image_temp = Image.open(path_img_temp)
image_temp2 = Image.open(path_img_temp2)
c1.title('c1')
c1.image(image_temp2)
c2.title('c2')
c2.image(image_temp)
c3.title('c3')
c3.image(image_temp)
c4.title('c4')
c4.image(image_temp2)

img = st.sidebar.selectbox(
    'Select Image',
    ('scan.npy', 'scan.npy')#,'amber.jpg', 'cat.png')
)

style_name = st.sidebar.selectbox(
    'Select Style',
    ('candy', 'mosaic', 'rain_princess', 'udnie')
)

## ARGUMENTS
SCAN_NAME = 'volume-covid19-A-0014'
ONLY_ONE_SLICE = 34
SEED_VALUE = 1 
TRESH_P = 20

# model= "saved_models/" + style_name + ".pth"
input_image = "images/content-images/" + img
output_image = "images/output-images/" + style_name + "-" + img

st.write('### Example scan:')
image_scan = np.load(input_image)
fig0 = plt.figure()
plt.imshow(np.rot90(normalize_scan(image_scan[...,34]),-1), cmap='gray')
plt.axis(False)
st.pyplot(fig0, width=400) 

load = st.button('CT load')
superpixels = st.button('superpixels')
test = st.button('test')





if load:
    scan, scan_mask, loader_lesions, texture = load_synthetic_lesions_scans_and__individual_lesions(SCAN_NAME)
    st.session_state.scan = normalize_scan(scan)
    
    st.session_state.scan_mask = scan_mask
    st.session_state.loader_lesions = loader_lesions
    st.session_state.texture = texture

    st.write(f'scan = {np.shape(scan), np.min(scan), np.max(scan)}')
    st.write(f'scan_mask = {np.shape(scan_mask), np.min(scan_mask), np.max(scan_mask)}')
    st.write(type(loader_lesions))
    st.image(np.rot90(st.session_state.scan[...,ONLY_ONE_SLICE],-1))

if test:
    st.write(type(loader_lesions))

if superpixels:
    superpixels_args = superpixels_applied(st.session_state.loader_lesions, ONLY_ONE_SLICE, SEED_VALUE=1)
    print(len(superpixels_args)) 
    img_lesions, mask_slic, boundaries_plot, segments, segments_sizes, coords_big, idx_mini_batch, numSegments = superpixels_args

    scan = st.session_state.scan
    scan_mask = st.session_state.scan_mask
    
    st.write(f'img_lesions = {np.shape(img_lesions), np.min(img_lesions), np.max(img_lesions)}')
    
    ct1, ct2, ct3, ct4 = st.beta_columns((1, 1, 1, 1))
    ct1.title('c1')
    fig1 = plt.figure()
    plt.imshow(np.rot90(scan[...,coords_big[-1]],-1), cmap='gray')
    plt.imshow(np.rot90(scan_mask[...,coords_big[-1]],-1), alpha=.3)
    ct1.pyplot(fig1)
    ct2.title('c2')
    fig2 = plt.figure()
    plt.imshow(np.rot90(scan[coords_big[0]-TRESH_P:coords_big[1]+TRESH_P,coords_big[2]-TRESH_P:coords_big[3]+TRESH_P,coords_big[-1]],-1), cmap='gray')
    plt.imshow(np.rot90(scan_mask[coords_big[0]-TRESH_P:coords_big[1]+TRESH_P,coords_big[2]-TRESH_P:coords_big[3]+TRESH_P,coords_big[-1]],-1), alpha=.3)
    ct2.pyplot(fig2)
    ct3.title('c3')
    fig3 = plt.figure()
    plt.imshow(np.rot90(img_lesions[0],-1),cmap='gray')
    ct3.pyplot(fig3)
    ct4.title('c4')
    fig4 = plt.figure()
    plt.imshow(np.rot90(boundaries_plot*mask_slic,-1), vmax=1, cmap='gray')
    ct4.pyplot(fig4)

# if clicked:
#     #model = style.load_model(model)
#     #style.stylize(model, input_image, output_image)

#     st.write('### Output image:')
#     image = Image.open(output_image)
#     st.image(image, width=400)