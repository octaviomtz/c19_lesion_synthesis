# python3 -m venv venv
# . venv/bin/activate
# pip install streamlit
# pip install torch torchvision
# streamlit run main.py

# TODO
# continue with replace_with_nCA but first replace the 
# default scan by one in housenfield units because slic
# doesn't work with ints and right now scan values are 0-255
from torch.functional import norm
import streamlit as st
from PIL import Image
from utils_load import (
    load_synthetic_lesions_scans_and__individual_lesions,
    superpixels_applied,
    load_only_original_scans,
    normalize_scan,
    from_scan_to_3channel,
    from_scan_to_3channel2,
    scale_rect_coords_and_compare_nodule_coords,
    superpixels2,
    figures_zoom_and_superpixels,
    load_synthetic_texture,
    replace_with_nCA,
)
from utils import (
    coords_min_max_2D
)
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

## FUNCTIONS


## ARGUMENTS
CANVA_HEIGHT = 400
CANVA_WIDTH = 400
SEED_VALUE = 1 
TRESH_P = 20
THRESH_DIST = 10

st.session_state['dist_coords_small'] = False

# st.set_page_config(layout="wide") 
# # import style
# # use https://github.com/andfanilo/streamlit-drawable-canvas

st.title('C19 Lesion Synthesis')

a1, a2 = st.beta_columns((1, 1))
st.session_state.ONLY_ONE_SLICE = a1.selectbox(
    'Select a slice',
    (34, 17, 18, 19, 20, 21, 22, 30, 31, 32, 33, 34, 35, 36))
scan_selection = a2.selectbox(
    'Select a scan with lesions',
    (14, 99))

SCAN_NAME = f'volume-covid19-A-{scan_selection:04d}'

# Load default scan and mask
@st.cache(suppress_st_warning=True)
def load_initially(SCAN_NAME):
    scan_3D = np.load(f'images/scans/{SCAN_NAME}.npy')
    scan_mask_3D = np.load(f'images/masks/{SCAN_NAME}.npz')
    scan_mask_3D = scan_mask_3D.f.arr_0
    slices_with_lesions = np.where(np.sum(scan_mask_3D, (0,1))>20)[0] 
    # st.write(slices_with_lesions, np.min(st.session_state['scan']),np.max(st.session_state['scan']))
    
    # fig = plt.figure()
    # plt.imshow(st.session_state['scan'])
    # st.pyplot(fig)
    texture = load_synthetic_texture()
    return scan_3D, scan_mask_3D, texture

@st.cache(suppress_st_warning=True)
def load_slice_initially(scan, scan_mask, ONLY_ONE_SLICE=st.session_state.ONLY_ONE_SLICE):
    scan_slice = scan[...,ONLY_ONE_SLICE]
    scan_mask = scan_mask[...,ONLY_ONE_SLICE]
    return scan_slice, scan_mask

## LOAD DEFAULT OR CHANGE SLICE 
scan_3D, scan_mask_3D, st.session_state['texture'] = load_initially(SCAN_NAME)
st.session_state['scan'], st.session_state['scan_mask'] = load_slice_initially(scan_3D, scan_mask_3D)
st.session_state['coords'] = coords_min_max_2D(st.session_state['scan_mask'])

c1, c2, c3, c4, _, _, _ = st.beta_columns((1, 1, 1, 1, 1, 1, 1))
load_only_scans = c1.button('CT scans_only')
load = c2.button('CT load')
superpixels = c3.button('superpixels')
test = st.button('test')
st.write(st.session_state['coords'])


img = st.sidebar.selectbox(
    'Select Image',
    ('scan.npy', 'scan.npy')#,'amber.jpg', 'cat.png')
)

# style_name = st.sidebar.selectbox(
#     'Select Style',
#     ('candy', 'mosaic', 'rain_princess', 'udnie')
# )

# model= "saved_models/" + style_name + ".pth"
# input_image = "images/content-images/" + img 
# output_image = "images/output-images/" + style_name + "-" + img

# st.write('### Example scan:')
# image_scan = np.load(input_image)
# fig0 = plt.figure()
# plt.imshow(np.rot90(normalize_scan(image_scan[...,34]),-1), cmap='gray')
# plt.axis(False)
# st.pyplot(fig0, width=400) 



 
## To save the scan as an image
# import imageio 
# aaa = np.rot90(normalize_scan(np.load(input_image)[...,ONLY_ONE_SLICE]),-1)
# aaa = np.repeat(np.expand_dims(aaa,-1),3,-1)
# aaa = aaa.astype(np.uint8)
# st.write(np.shape(aaa))
# imageio.imwrite('images/content-images/aaa.png', aaa)
# aaa.save('images/content-images/aaa.png')



if load_only_scans:
    scan, scan_mask = load_only_original_scans(SCAN_NAME)
    st.session_state.scan = scan 
    st.session_state.scan_mask = scan_mask
    scan_plot = from_scan_to_3channel(scan, slice= st.session_state.ONLY_ONE_SLICE, normalize=True, rotate=-1)

    fig_only_scans = plt.figure()
    plt.imshow(scan_plot)
    st.pyplot(fig_only_scans)


if load:
    scan, scan_mask, loader_lesions, texture = load_synthetic_lesions_scans_and__individual_lesions(SCAN_NAME)
    st.session_state.scan = normalize_scan(scan)    
    st.session_state.scan_mask = scan_mask
    st.session_state.loader_lesions = loader_lesions
    st.session_state.texture = texture

    st.write(f'scan = {np.shape(scan), np.min(scan), np.max(scan)}')
    st.write(f'scan_mask = {np.shape(scan_mask), np.min(scan_mask), np.max(scan_mask)}')
    st.write(type(loader_lesions))
    st.image(np.rot90(st.session_state.scan[...,st.session_state.ONLY_ONE_SLICE],-1))
    
    # np.save('images/scans/scan.npy', scan_mask)
    # np.savez_compressed('images/scans/scan_mask.npz', scan_mask)

if superpixels: 
    superpixels_args = superpixels_applied(st.session_state.loader_lesions, st.session_state.ONLY_ONE_SLICE, SEED_VALUE=1)
    print(len(superpixels_args)) 
    img_lesions, mask_slic, boundaries_plot, segments, segments_sizes, coords_big, idx_mini_batch, numSegments = superpixels_args
    st.write(coords_big)
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

    st.write(coords_big)
    # canvas_result.realtime_update = False

# if clicked:
#     #model = style.load_model(model)
#     #style.stylize(model, input_image, output_image)

#     st.write('### Output image:')
#     image = Image.open(output_image)
#     st.image(image, width=400)

########## DRAWABLE

# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ") 
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
# drawing_mode = st.sidebar.selectbox(
#     "Drawing tool:", ("rect","freedraw", "line", "circle", "transform")
# )
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    # background_image=Image.open(bg_image) if bg_image else None,
    # background_image=Image.open('images/content-images/scan_default.png'),
    background_image= from_scan_to_3channel(st.session_state['scan'], slice= st.session_state.ONLY_ONE_SLICE, normalize=False),
    update_streamlit=realtime_update,
    height= CANVA_HEIGHT,
    width = CANVA_WIDTH,
    # drawing_mode=drawing_mode,
    drawing_mode= 'rect', 
    key="canvas",
)




# Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)
if len(canvas_result.json_data["objects"]) >= 1:
    if len(canvas_result.json_data["objects"]) == 1:
        canvas_result.drawing_mode = 'transform'
        # canvas_result.json_data["objects"].pop(0)   
    # st.write(canvas_result.json_data)
    res = canvas_result.json_data["objects"][0]
    # canva_shape0, canva_shape1 = np.shape(st.session_state['scan'])
    # scale_y = canva_shape0/CANVA_HEIGHT
    # scale_x = canva_shape1/CANVA_WIDTH
    coords_scaled, dist_coords, st.session_state['dist_coords_small'] = scale_rect_coords_and_compare_nodule_coords(st.session_state['scan'], canvas_result.json_data["objects"][0], st.session_state['coords'], CANVA_HEIGHT=CANVA_HEIGHT, CANVA_WIDTH=CANVA_WIDTH)
    st.write(coords_scaled, st.session_state['coords'], dist_coords)
    st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))

if st.session_state['dist_coords_small']:  
    dist_small1, dist_small2, dist_small3 = st.beta_columns((1, 1, 1))
    mask_slic, boundaries, segments = superpixels2(st.session_state['scan'], st.session_state['scan_mask'])
    fig_zoom1, fig_zoom2, fig_zoom3 = figures_zoom_and_superpixels(st.session_state['scan'], st.session_state['scan_mask'], st.session_state['coords'], boundaries)
    dist_small1.pyplot(fig_zoom1)
    dist_small2.pyplot(fig_zoom2)
    dist_small3.pyplot(fig_zoom3)

if test:
    imB1, imB2 = st.beta_columns((1, 1))
    figB1 = plt.figure()
    imgB1 = from_scan_to_3channel(st.session_state['scan'], slice= st.session_state.ONLY_ONE_SLICE, normalize=False)
    st.write(np.shape(imgB1))
    plt.imshow(imgB1)
    plt.imshow(st.session_state['scan_mask'], alpha=.3)
    imB1.pyplot(figB1)
    
    arrays_sequence, images, seq_idx = replace_with_nCA(st.session_state['scan'], SCAN_NAME, st.session_state.ONLY_ONE_SLICE, st.session_state['texture'])
    st.write(f'arrays_sequence={np.shape(arrays_sequence)}, images={np.shape(images)}, decreasing_sequence={len(seq_idx)},{seq_idx[-5:]}')

    fig_syn = plt.figure()
    plt.imshow(arrays_sequence[10], cmap='gray')
    plt.axis('off')
    plt.savefig('images/output-images/temp_syn.png')
    st.pyplot(fig_syn)