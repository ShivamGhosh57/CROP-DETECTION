import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# --- App Configuration ---
st.set_page_config(
    page_title="Crop Pest & Disease Detection",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Load The Model ---
# Cache the model loading to prevent reloading on every interaction
@st.cache_resource
def load_pest_model():
    """
    Loads the pre-trained crop pest detection model.
    """
    try:
        # Load the model with custom objects if it contains custom layers/activations
        model = load_model('crop_pest_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure the 'crop_pest_model.h5' file is in the same directory as this script.")
        return None

model = load_pest_model()

# --- Class Names ---
# IMPORTANT: This list must match the training data's class indices.
CLASS_NAMES = [
    'Cashew anthracnose', 'Cashew gumosis', 'Cashew healthy', 'Cashew leaf miner', 'Cashew red rust',
    'Cassava bacterial blight', 'Cassava brown spot', 'Cassava green mite', 'Cassava healthy', 'Cassava mosaic',
    'Maize fall armyworm', 'Maize grasshoper', 'Maize healthy', 'Maize leaf beetle', 'Maize leaf blight',
    'Maize leaf spot', 'Maize streak virus', 'Tomato healthy', 'Tomato leaf blight', 'Tomato leaf curl',
    'Tomato septoria leaf spot', 'Tomato verticulium wilt'
]

# --- Image Preprocessing ---
def preprocess_image(image):
    """
    Preprocesses the user-uploaded image to match the model's input requirements.
    """
    # Ensure image is in RGB format (3 channels)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Resize the image to the target size (150x150)
    img = image.resize((150, 150))
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Expand dimensions to create a batch of 1
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values
    img_array = img_array / 255.0
    return img_array


# --- Main Application UI ---
st.title("🌿 Crop Pest & Disease Detection")
st.markdown("Upload an image of a crop leaf, and the model will predict the pest or disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True) # Updated parameter
    st.write("")

    # Predict button
    if st.button('Detect Pest/Disease'):
        with st.spinner('Analyzing the image...'):
            # Preprocess the image
            processed_image = preprocess_image(image)

            # Make a prediction
            try:
                prediction = model.predict(processed_image)
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                confidence = np.max(prediction) * 100

                st.success(f"**Prediction:** {predicted_class_name}")
                st.info(f"**Confidence:** {confidence:.2f}%")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

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
