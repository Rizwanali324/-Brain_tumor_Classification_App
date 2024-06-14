import os 
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

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

# Define the directory containing subfolders with class names
base_dir = 'test_images'

# Get the list of subfolders (each representing a class)
class_folders = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]

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
    st.title("Brain Tumor Classification App")

    st.write("""
    This app classifies brain MRI images into one of the following categories:
    The model is trained on the Brain Tumor MRI Dataset, which can be found [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
    """)

    display_first_images(class_folders)

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image.', width=350)

        # Prediction button
        if st.button('Predict'):
            # Get predictions
            predictions = predict(image)
            predicted_class_idx = np.argmax(predictions)

            # Define class names (ensure it matches your model's output)
            class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

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
