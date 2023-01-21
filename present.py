import streamlit as st
import requests
import tensorflow as tf
import re
import string
import numpy as np
import pickle as pkl
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import layers,Model

vgg16 = VGG16(weights='imagenet')
fe = Model(inputs = vgg16.input, outputs = vgg16.layers[-2].output)

def feature_extract(image):
    img = Image.open(image).resize((224,224))
    img = img_to_array(img)
    img = img[None,:]
    img = preprocess_input(img)
    return fe.predict(img)

max_len = 38
vocab_size = 8831
def base_model():
    input1 = layers.Input(shape=(4096,))
    fe1 = layers.Dropout(0.5)(input1)
    fe2 = layers.Dense(256, activation= 'relu')(fe1)

    input2 = layers.Input(shape=(max_len,))
    enc1 = layers.Embedding(vocab_size,
                            256,
                            mask_zero=True
                            )(input2)
    enc2 = layers.Dropout(0.5)(enc1)
    enc3 = layers.LSTM(256)(enc2)

    decoder1 = layers.add([fe2, enc3])
    decoder2 = layers.Dense(256, activation = 'relu')(decoder1)
    output = layers.Dense(vocab_size, activation = 'softmax')(decoder2)

    model = Model(inputs = [input1, input2],
                    outputs = output)

    model.compile(
            loss = 'categorical_crossentropy',
            optimizer = 'adam'
            )
    return model

def predict(image_name, feature_extract, model, tokenizer, max_len, start_token = 'startseq', end_token = 'endseq'):
    output = start_token
    img_feature = feature_extract(image_name)
    for i in range(max_len):
        sequence = tokenizer.texts_to_sequences([output])[0]
        sequence = pad_sequences([sequence], maxlen=max_len)
        yhat = model.predict([img_feature, sequence])
        idx = np.argmax(yhat)
        if idx in tokenizer.index_word:
            token = tokenizer.index_word[idx]
            output += ' ' + token
            if token == end_token:
                break
        else:
            break
    return output

@st.cache(allow_output_mutation=True)
def load_session():
    return requests.Session()

def main():
    st.set_page_config(
        page_title="Image captioning",
        page_icon=":star:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title(":newspaper: Image captioning")
    sess = load_session()

    uploaded_images = st.file_uploader("Choose an image: ", accept_multiple_files=True)
    model = base_model()
    with open('tokenizer.pkl', 'rb+') as f:
        tokenizer = pkl.load(f)
            
    col1, col2 = st.columns([6, 4])
    with col2:
        st.image(f"model.png", width=700, caption='Model base on VGG16 + LSTM')

    with col1:
        if len(uploaded_images) != 0:
            st.image(uploaded_images, caption = [image.name for image in uploaded_images])

    button = st.button("Predict")

    st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )
    if button:
        with st.spinner("Predicting..."):
            if len(uploaded_images) == 0:
                st.markdown('Please upload (an) images')
            else:
                for image in uploaded_images:
                    caption = predict(image, feature_extract, model, tokenizer, max_len)
                    st.markdown(image.name)
                    st.markdown(caption)


if __name__ == "__main__":
    main()