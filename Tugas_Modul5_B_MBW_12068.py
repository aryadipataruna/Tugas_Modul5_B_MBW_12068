import streamlit as st
import numpy as np
import pickle
from PIL import Image
import os

# Model path
model_path = 'best_model.pkl'

# Load the model
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        # Class labels
        class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # Image preprocessing function
        def preprocess_image(image):
            image = image.resize((28, 28))  # Resize to 28x28
            image = image.convert('L')  # Convert to grayscale
            image_array = np.array(image) / 255.0  # Normalize pixel values
            image_array = image_array.flatten().reshape(1, -1)  # Flatten and reshape for the model
            return image_array

        # Streamlit app
        st.title("Fashion MNIST Image Classifier")
        st.write("Unggah gambar item fashion (misalnya sepatu, tas, baju), dan model akan memprediksi kelasnya.")

        # File uploader
        uploaded_files = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        # Sidebar for prediction
        with st.sidebar:
            st.write("## Navigator")
            predict_button = st.button("Predict")

        if uploaded_files and predict_button:
            st.write("## Hasil Prediksi")

            for uploaded_file in uploaded_files:
                try:
                    # Process and predict
                    image = Image.open(uploaded_file)
                    processed_image = preprocess_image(image)
                    predictions = model.predict_proba(processed_image)
                    predicted_class = np.argmax(predictions)
                    confidence = np.max(predictions) * 100

                    # Display results
                    st.image(image, caption=f"Gambar: {uploaded_file.name}", use_column_width=True)
                    st.write(f"**Nama File:** {uploaded_file.name}")
                    st.write(f"Kelas Prediksi: **{class_name[predicted_class]}**")
                    st.write(f"Confidence: **{confidence:.2f}%**")
                    st.write("---")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

else:
    st.error("File model tidak ditemukan.")
