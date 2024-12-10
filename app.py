import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
import streamlit as st

# Definisi model
def create_model():
    model = Sequential([
        EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        GlobalAveragePooling2D(),
        Dense(3, activation='softmax')  # Jumlah kelas
    ])
    return model

# Load model
weights_path = 'my_model_weights-v1.weights.h5'  # Sesuaikan path jika berbeda
model = create_model()

try:
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    st.success("Bobot model berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat bobot: {e}")

# Fungsi prediksi
def predict_image(image_path, model):
    from tensorflow.keras.preprocessing import image
    import numpy as np

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    prediction = model.predict(img_array)
    return prediction

# Unggah gambar
uploaded_file = st.file_uploader("Unggah gambar daun (.jpg, .png)", type=["jpg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    prediction = predict_image("uploaded_image.jpg", model)
    st.write(f"Hasil prediksi: {prediction}")
