import os 
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Define the directory containing subfolders with class names
base_dir = 'test_images'

# Function to preprocess the image
def preprocess_image(image):
    size = (256, 256)
    image = image.resize(size)
    image = image.convert('L')  # Convert to grayscale
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to make predictions
def predict(image):
    image_array = preprocess_image(image)
    predictions = model.predict(image_array)
    return predictions

# Function to load and display the first image from each class folder
def display_first_images(class_folders):
    num_folders = len(class_folders)
    fig, axes = plt.subplots(1, num_folders, figsize=(10, 3))

    for i, folder in enumerate(class_folders):
        class_name = os.path.basename(folder)
        images = os.listdir(folder)
        image_path = os.path.join(folder, images[0])  # Get the first image in the folder
        image = Image.open(image_path)
        image = image.resize((150, 150))  # Resize the image for display
        axes[i].imshow(image)
        axes[i].set_title(class_name)
        axes[i].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

# Define Streamlit app
def main():
    global base_dir  # Ensure base_dir is global if you intend to access it inside functions
    global model  # Ensure model is global if you intend to access it inside functions
    
    # Streamlit app setup
    st.set_page_config(page_title="Brain Tumor Classification", page_icon=":brain:", layout='wide', initial_sidebar_state='expanded')
    st.sidebar.markdown("# aibytech")
    
    st.sidebar.image('logo.jpg', width=200)
    st.title("Brain Tumor Classification App")

    st.write("""
    This app classifies brain MRI images into one of the following categories:
    The model is trained on the Brain Tumor MRI Dataset, which can be found [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
    """)

    # Get the list of subfolders (each representing a class)
    class_folders = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]

    display_first_images(class_folders)

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image and prediction button in a column layout
        uploaded_image = Image.open(uploaded_file)
        col1, col2 = st.columns([2, 1])  # Adjust the width ratio as needed

        with col1:
            st.image(uploaded_image, caption='Uploaded MRI Image.', width=350)
        
        with col2:
            # Prediction button
            if st.button('Predict'):
                # Get predictions
                predictions = predict(uploaded_image)
                predicted_class_idx = np.argmax(predictions)

                # Define class names (ensure it matches your model's output)
                class_names = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

                # Check if predicted_class_idx is within bounds
                if 0 <= predicted_class_idx < len(class_names):
                    predicted_class = class_names[predicted_class_idx]
                    st.write(f"Prediction: {predicted_class}")

                    # Display caution warning
                    st.warning("""
                    Caution: This app provides predictions based on machine learning models. Use the results as a supplementary tool and consult medical professionals for definitive diagnosis.
                    """)

                else:
                    st.write("Invalid prediction index.")

if __name__ == "__main__":
    main()
