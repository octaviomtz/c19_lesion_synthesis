# python3 -m venv venv
# . venv/bin/activate
# pip install streamlit
# pip install torch torchvision
# streamlit run main.py

# TODO
# make sure superpixels2 produce the right boundaries
from torch.functional import norm
import streamlit as st
from PIL import Image
from utils_load import (
    load_synthetic_lesions_scans_and__individual_lesions,
    superpixels_applied,
    load_only_original_scans,
    normalize_scan,
    original_slice_to_cannonical_view,
    from_scan_to_3channel2,
    scale_rect_coords_and_compare_nodule_coords,
    superpixels2,
    figures_zoom_and_superpixels,
    load_synthetic_texture,
    replace_with_nCA,
    open_local_gif,
)
from utils import (
    coords_min_max_2D
)
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from PIL import Image
import streamlit as st
import imageio
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

a1, a2, a3, a4 = st.columns((1, 1, 1, 1))
st.session_state.ONLY_ONE_SLICE = a1.selectbox(
    'Select a slice',
    (34, 17, 18, 19, 20, 21, 22, 30, 31, 32, 33, 34, 35, 36))
scan_selection = a2.selectbox(
    'Select a scan with lesions',
    (14, 99))
test = a3.button('test')

SCAN_NAME = f'volume-covid19-A-{scan_selection:04d}'

# Load default scan and mask
@st.cache(suppress_st_warning=True)
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

@st.cache(suppress_st_warning=True)
def load_slice_initially(scan, scan_mask, ONLY_ONE_SLICE=st.session_state.ONLY_ONE_SLICE):
    scan_slice = scan[...,ONLY_ONE_SLICE]
    scan_mask = scan_mask[...,ONLY_ONE_SLICE]
    return scan_slice, scan_mask

## LOAD DEFAULT OR CHANGE SLICE 
scan_3D, scan_mask_3D, st.session_state['texture'] = load_initially(SCAN_NAME)
st.session_state['scan'], st.session_state['scan_mask'] = load_slice_initially(scan_3D, scan_mask_3D)
st.session_state['coords'] = coords_min_max_2D(st.session_state['scan_mask'])

c1, c2, c3, c4, _, _, _ = st.columns((1, 1, 1, 1, 1, 1, 1))


# st.write(np.min(st.session_state['scan']), np.max(st.session_state['scan']), st.session_state['coords'])

img = st.sidebar.selectbox(
    'Select Image',
    ('scan.npy', 'scan.npy')#,'amber.jpg', 'cat.png')
)

## To save the scan as an image
# import imageio 
# aaa = np.rot90(normalize_scan(np.load(input_image)[...,ONLY_ONE_SLICE]),-1)
# aaa = np.repeat(np.expand_dims(aaa,-1),3,-1)
# aaa = aaa.astype(np.uint8)
# st.write(np.shape(aaa))
# imageio.imwrite('images/content-images/aaa.png', aaa)
# aaa.save('images/content-images/aaa.png')

########## DRAWABLE

# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ") 
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("rect","freedraw", "line", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

def rect_or_transform(rect = True):
    rect_or_trans = 'transform'
    if rect:
        rect_or_trans = 'rect'
    return rect_or_trans
    

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    # background_image=Image.open(bg_image) if bg_image else None,
    # background_image=Image.open('images/content-images/scan_default.png'),
    background_image= from_scan_to_3channel2(original_slice_to_cannonical_view(st.session_state['scan'])),
    update_streamlit=realtime_update,
    height= CANVA_HEIGHT,
    width = CANVA_WIDTH,
    drawing_mode=drawing_mode,
    # drawing_mode= 'rect', 
    # drawing_mode= rect_or_transform(), 
    key="canvas",
)


# with st.form("my_form",clear_on_submit=True):
#     st.write("Inside the form")
#     submitted = st.form_submit_button("Submit")
#     slider_val = st.slider("Form slider")
#     checkbox_val = st.checkbox("Form checkbox")
#     if submitted:
#         st.write("slider", slider_val, "checkbox", checkbox_val)

# Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)
if len(canvas_result.json_data["objects"]) >= 1:
    rect_or_transform(rect = False)
    if len(canvas_result.json_data["objects"]) == 1:
        canvas_result.drawing_mode = 'line'
        # canvas_result.json_data["objects"].pop(0)   
    if  len(canvas_result.json_data["objects"]) == 3:
        canvas_result.background_image = from_scan_to_3channel2(original_slice_to_cannonical_view(st.session_state['scan']))
        # canvas_result.json_data.clear()
        canvas_result.height=20

    res = canvas_result.json_data["objects"][0]
    coords_scaled, dist_coords, st.session_state['dist_coords_small'] = scale_rect_coords_and_compare_nodule_coords(st.session_state['scan'], canvas_result.json_data["objects"][0], st.session_state['coords'], CANVA_HEIGHT=CANVA_HEIGHT, CANVA_WIDTH=CANVA_WIDTH)
    # st.write(coords_scaled, st.session_state['coords'], dist_coords)
    df_coords = pd.DataFrame(pd.json_normalize(canvas_result.json_data["objects"]))
    df_coords = df_coords[['left', 'top', 'width', 'height']]
    st.dataframe(df_coords)

if st.session_state['dist_coords_small']:  
    dist_small1, dist_small2, dist_small3 = st.columns((1, 1, 1))
    coords = st.session_state['coords']
    scan_norm = original_slice_to_cannonical_view(st.session_state['scan'], to_0_255=False)
    mask_slic, boundaries, segments = superpixels2(scan_norm[coords[0]: coords[1], coords[2]:coords[3]], st.session_state['scan_mask'][coords[0]: coords[1], coords[2]:coords[3]])
    fig_zoom1, fig_zoom2, fig_zoom3 = figures_zoom_and_superpixels(original_slice_to_cannonical_view(st.session_state['scan']), st.session_state['scan_mask'], st.session_state['coords'], boundaries)
    dist_small1.pyplot(fig_zoom1)
    dist_small2.pyplot(fig_zoom2)
    dist_small3.pyplot(fig_zoom3)
    # figXX, ax = plt.subplots(1,2)
    # ax[0].imshow(scan_norm[coords[0]: coords[1], coords[2]:coords[3]])
    # ax[1].imshow(boundaries[coords[0]: coords[1], coords[2]:coords[3]])
    # st.pyplot(figXX)
    # st.write(np.unique(st.session_state['scan_mask']))

if test:
    with st.spinner('Replacing lesion with cellular automata....'):
        slider_gen = st.slider("synthesis generation", min_value=0, max_value=40, value=5)
        imB1, imB2 = st.columns((1, 1))  
        scan_norm = original_slice_to_cannonical_view(st.session_state['scan'], to_0_255=False, rotate=False, fliplr=False)
        arrays_sequence, seq_idx = replace_with_nCA(scan_norm, SCAN_NAME, st.session_state.ONLY_ONE_SLICE, st.session_state['texture'])
        # st.write(f'arrays_sequence={np.shape(arrays_sequence)}, decreasing_sequence={len(seq_idx)},{seq_idx[-5:]}')
        st.write(f'arrays_sequence={np.shape(arrays_sequence)}')  
        fig_syn = plt.figure()
        text_idx = slider_gen
        plt.text(20,20,text_idx, c='#7ccfa7', fontsize=20)
        plt.imshow(original_slice_to_cannonical_view(arrays_sequence[text_idx], normalize_scan_hounsfield = False, to_0_255=False), cmap='gray')
        plt.axis('off')
        imB1.pyplot(fig_syn)

        fig_syn = plt.figure()
        plt.text(5,10,text_idx, c='#7ccfa7', fontsize=20)
        coords = st.session_state['coords']
        plt.imshow(original_slice_to_cannonical_view(arrays_sequence[text_idx], normalize_scan_hounsfield = False, to_0_255=False)[coords[0]: coords[1], coords[2]:coords[3]], cmap='gray')
        plt.axis('off')
        imB2.pyplot(fig_syn)

        imS1, imS2 = st.columns((10, 1))
        gif = open_local_gif(location=f'images/for_gifs/{SCAN_NAME}_{st.session_state["ONLY_ONE_SLICE"]}.gif')
        imS1.markdown(
            f'<img src="data:image/gif;base64,{gif}" alt="gif" width="80%">',
            unsafe_allow_html=True,
        )
    
    