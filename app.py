import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image

# Fungsi untuk memuat model
@st.cache_resource
def load_model(weights_path):
    model = Sequential([
        EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3)),
        GlobalAveragePooling2D(),
        Dense(3, activation='softmax')  # Sesuaikan jumlah kelas
    ])
    model.load_weights(weights_path)
    return model

# Fungsi prediksi dengan model
def predict_image(image_path, model):
    class_labels = ['Anthracnose', 'Bacterial Canker', 'Healthy']  # Sesuaikan kelas dengan data Anda
    
    # Preprocessing gambar
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class_index]
    
    return predicted_label, prediction[0]

# Path ke bobot model
weights_path = 'my_model_weights-v1.weights.h5'  # Sesuaikan dengan lokasi file Anda

# Memuat model
try:
    model = load_model(weights_path)
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")

# Antarmuka Streamlit
st.title("Aplikasi Prediksi Penyakit Daun Mangga")
st.write("Unggah gambar daun mangga untuk memprediksi jenis penyakit.")

# Unggah file
uploaded_file = st.file_uploader("Unggah gambar daun (.jpg, .png)", type=["jpg", "png"])

if uploaded_file:
    # Simpan file sementara
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Tampilkan gambar yang diunggah
    st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
    
    # Prediksi gambar
    try:
        predicted_label, prediction_probabilities = predict_image("uploaded_image.jpg", model)
        st.write(f"**Hasil Prediksi:** {predicted_label}")
        st.write(f"**Probabilitas Tiap Kelas:** {prediction_probabilities}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
