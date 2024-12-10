import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# Konfigurasi model
@st.cache_resource  # Cache model untuk mempercepat pengujian
def build_and_load_model(weights_path):
    try:
        # Definisikan model
        model = Sequential([
            EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3)),
            GlobalAveragePooling2D(),
            Dense(3, activation='softmax')  # Sesuaikan jumlah kelas
        ])
        # Memuat bobot model
        model.load_weights(weights_path)
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

# Fungsi prediksi
def predict_image(uploaded_image, model, class_labels):
    try:
        # Baca dan proses gambar
        img = Image.open(uploaded_image)
        img = img.resize((224, 224))  # Sesuaikan ukuran gambar
        img_array = np.array(img)
        if img_array.shape[-1] != 3:
            st.error("Gambar tidak memiliki 3 channel (RGB). Harap unggah gambar yang valid.")
            return None, None
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Prediksi menggunakan model
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
        confidence = np.max(prediction)

        return predicted_class_label, confidence
    except Exception as e:
        st.error(f"Error saat memproses gambar: {e}")
        return None, None

# Streamlit UI
st.title("Prediksi Penyakit Daun Mangga")
st.write("Unggah gambar daun mangga untuk memprediksi jenis penyakitnya.")

# Definisi label kelas
class_labels = ['Anthracnose', 'Bacterial Canker', 'Healthy']

# Path ke bobot model
weights_path = 'my_model_weights-v1.weights.h5'  # Sesuaikan path jika berbeda

# Load model
model = build_and_load_model(weights_path)

# Cek apakah model berhasil dimuat
if model:
    # Unggah gambar
    uploaded_file = st.file_uploader("Unggah gambar daun (.jpg, .png)", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
        st.write("Memproses gambar...")

        # Prediksi
        label, confidence = predict_image(uploaded_file, model, class_labels)

        if label is not None:
            st.write(f"**Prediksi:** {label}")
            st.write(f"**Kepercayaan:** {confidence:.2f}")
else:
    st.error("Model gagal dimuat. Pastikan path bobot model benar.")
