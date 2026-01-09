import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import gdown

# --- App Configuration ---
st.set_page_config(
    page_title="Crop Pest & Disease Detection",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Model File ID ---
# This is the ID extracted from your specific Google Drive link
GOOGLE_DRIVE_FILE_ID = '1GbPoQBNWAFs-Cg9XINHJDIlyKob4TCZD'

# --- Load The Models ---
@st.cache_resource
def load_pest_model():
    """
    Downloads the model from Google Drive if not present, then loads it.
    """
    model_path = 'crop_pest_model.h5'
    
    # Check if model exists locally; if not, download it
    if not os.path.exists(model_path):
        
        try:
            # Construct the download URL for gdown
            url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
            gdown.download(url, model_path, quiet=False)
            
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None

    try:
        # Load the model
        # compile=False is safer/faster for inference-only
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

vision_model = load_pest_model()

# --- Class Names ---
CLASS_NAMES = [
    'Cashew anthracnose', 'Cashew gumosis', 'Cashew healthy', 'Cashew leaf miner', 'Cashew red rust',
    'Cassava bacterial blight', 'Cassava brown spot', 'Cassava green mite', 'Cassava healthy', 'Cassava mosaic',
    'Maize fall armyworm', 'Maize grasshoper', 'Maize healthy', 'Maize leaf beetle', 'Maize leaf blight',
    'Maize leaf spot', 'Maize streak virus', 'Tomato healthy', 'Tomato leaf blight', 'Tomato leaf curl',
    'Tomato septoria leaf spot', 'Tomato verticulium wilt'
]

# --- Helper Functions ---
def preprocess_image(image):
    """Preprocesses the image for the model."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = image.resize((150, 150))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# --- UI Layout ---
st.title("🌿 Crop Pest & Disease Detection")
st.markdown("Upload a crop leaf image for instant disease classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Using use_column_width for compatibility
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Crop Health'):
        if vision_model is None:
            st.error("Model failed to load. Please check internet connection or Drive Link.")
        else:
            with st.spinner('Analyzing...'):
                processed_image = preprocess_image(image)
                try:
                    prediction = vision_model.predict(processed_image)
                    idx = np.argmax(prediction, axis=1)[0]
                    pred_name = CLASS_NAMES[idx]
                    confidence = np.max(prediction) * 100

                    st.success(f"**Prediction:** {pred_name}")
                    st.info(f"**Confidence:** {confidence:.2f}%")

                except Exception as e:
                    st.error(f"Prediction Error: {e}")

# --- Sidebar Information ---
st.sidebar.title("About")
st.sidebar.info(
    "This application uses a deep learning model (ResNet50) to identify various crop pests "
    "and diseases from an uploaded image. It was trained on a dataset of 22 different classes."
)
st.sidebar.title("How to Use")
st.sidebar.markdown(
    "1. **Upload Image:** Click the 'Browse files' button to upload a leaf image.\n"
    "2. **Detect:** Click the 'Detect Pest/Disease' button.\n"
    "3. **View Result:** The predicted pest/disease and confidence level will be displayed."
)
st.sidebar.markdown("---")
st.sidebar.write("Developed based on the model from the notebook.")
