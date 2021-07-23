# python3 -m venv venv
# . venv/bin/activate
# pip install streamlit
# pip install torch torchvision
# streamlit run main.py

import streamlit as st
from PIL import Image
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
c4.image(image_temp)

img = st.sidebar.selectbox(
    'Select Image',
    ('amber.jpg', 'cat.png')
)

style_name = st.sidebar.selectbox(
    'Select Style',
    ('candy', 'mosaic', 'rain_princess', 'udnie')
)


# model= "saved_models/" + style_name + ".pth"
input_image = "images/content-images/" + img
output_image = "images/output-images/" + style_name + "-" + img

st.write('### Source image:')
image = Image.open(input_image)
st.image(image, width=400) # image: numpy array

clicked = st.button('Stylize')

if clicked:
    #model = style.load_model(model)
    #style.stylize(model, input_image, output_image)

    st.write('### Output image:')
    image = Image.open(output_image)
    st.image(image, width=400)