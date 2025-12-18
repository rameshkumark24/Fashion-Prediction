import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st 
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="Fashion Recommender", page_icon="👗", layout="wide")

st.title('👗 AI Fashion Recommendation System')
st.markdown("Upload an image of a clothing item, and the AI will find similar products from our database.")

# --- 1. Load Data & Model (Cached for Speed) ---
@st.cache_resource
def load_features_and_filenames():
    # Load your pre-computed features and filenames
    # Ensure these files are in the same folder as app.py
    try:
        feature_list = pkl.load(open('Images_features.pkl', 'rb'))
        filenames = pkl.load(open('filenames.pkl', 'rb'))
        return feature_list, filenames
    except FileNotFoundError:
        st.error("❌ Error: 'Images_features.pkl' or 'filenames.pkl' not found. Please check your files.")
        return None, None

@st.cache_resource
def load_model():
    # Load ResNet50 only once
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = tf.keras.models.Sequential([
        base_model,
        GlobalMaxPool2D()
    ])
    return model

# Initialize
with st.spinner("Loading AI Models..."):
    Image_features, filenames = load_features_and_filenames()
    model = load_model()

if Image_features is not None and model is not None:
    # Train NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(Image_features)

    # --- 2. Feature Extraction Function ---
    def extract_features_from_images(image_path, model):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_expand_dim = np.expand_dims(img_array, axis=0)
        img_preprocess = preprocess_input(img_expand_dim)
        result = model.predict(img_preprocess).flatten()
        norm_result = result / norm(result)
        return norm_result

    # --- 3. UI & Logic ---
    col_input, col_results = st.columns([1, 2])
    
    with col_input:
        st.subheader("1. Upload Image")
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Save and display uploaded file
        os.makedirs('upload', exist_ok=True)
        saved_path = os.path.join('upload', uploaded_file.name)
        with open(saved_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        with col_input:
            st.image(saved_path, caption="Uploaded Image", use_column_width=True)

        # Generate Recommendations
        with st.spinner("Analyzing style..."):
            input_img_features = extract_features_from_images(saved_path, model)
            distances, indices = neighbors.kneighbors([input_img_features])

        with col_results:
            st.subheader("2. Recommended Items")
            
            # Create columns for the 5 recommendations (skipping index 0 as it's the query itself usually)
            cols = st.columns(5)
            
            # Loop through recommendations
            # Note: indices[0][0] might be the uploaded image itself if it exists in DB, so we usually take 1:6
            for i, col in enumerate(cols):
                try:
                    # Get the filename from your list
                    recommended_file = filenames[indices[0][i+1]]
                    
                    # ⚠️ IMPORTANT: Fix path if running on different machine
                    # If filenames contains full paths like "C:/Users/...", you might need to fix them here
                    # Example: recommended_file = "data/images/" + os.path.basename(recommended_file)
                    
                    col.image(recommended_file, caption=f"Sim: {distances[0][i+1]:.2f}", use_column_width=True)
                except Exception as e:
                    col.error(f"Image not found")
