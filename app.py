import streamlit as st
from tensorflow import keras
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image


st.title('ResNet50')

st.header('Please upload an image')

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

model = load_model("mnist_resnet50.h5")

if file is not None:
    image = load_img(file, color_mode='grayscale' if model.input_shape[-1] == 1 else 'rgb', target_size=(28, 28))
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0

    if model.input_shape[-1] == 1 and image.shape[-1] == 3:
        image = image.mean(axis=-1, keepdims=True)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    st.write("## Prediction class")
    st.write(predicted_class[0])