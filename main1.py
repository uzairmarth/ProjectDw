import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import wikipedia
import pydeck as pdk
import pandas as pd
import matplotlib.pyplot as plt

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Dictionary with animal information
animal_info = {
    "Elephent": "Sumatran elephants are critically endangered due to habitat loss and poaching.",
    "Tiger": "Sunda tigers are a subspecies of tigers found primarily in the Sunda Islands of Indonesia.",
    "Rhino": "Black rhinos are critically endangered and are native to eastern and southern Africa."
}

# Streamlit app
st.title("Animal Detection Dashboard")

# Function to get species info from Wikipedia
def get_species_info(species):
    try:
        summary = wikipedia.summary(species, sentences=2)
    except:
        summary = "No information available."
    return summary

# Option to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
detected_species = None
confidence_scores = []

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Resize and process the image
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip().split(" ")[1]
    confidence_score = prediction[0][index]
    confidence_scores.append(confidence_score)
    detected_species = class_name

    # Get animal information
    if class_name in animal_info:
        info = animal_info[class_name]
    else:
        info = "No information available."

    # Display results
    st.write(f"Class: {class_name}  Confidence: {confidence_score:.2f}")
    st.write(f"Information: {info}")

# Option to use the camera
if st.button('Use Camera') and cv2.VideoCapture(0).isOpened():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    detection_placeholder = st.empty()
    info_placeholder = st.empty()
    time_series = []
    confidence_scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and process the frame
        image = Image.fromarray(frame)
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Predict using the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip().split(" ")[1]
        confidence_score = prediction[0][index]
        confidence_scores.append(confidence_score)
        detected_species = class_name

        # Get animal information
        if class_name in animal_info:
            info = animal_info[class_name]
        else:
            info = "No information available."

        # Update the frame and detection information
        stframe.image(frame, channels="BGR")
        detection_placeholder.text(f"Class: {class_name}  Confidence: {confidence_score:.2f}")
        info_placeholder.text(f"Information: {info}")

        # Update time series and confidence scores
        time_series.append(len(time_series))

    cap.release()

    # Plot the real-time confidence scores
    st.subheader("Real-Time Confidence Score")
    fig, ax = plt.subplots()
    ax.plot(time_series, confidence_scores, label="Confidence %")
    ax.set_xlabel("Time")
    ax.set_ylabel("Confidence")
    ax.legend()
    st.pyplot(fig)

else:
    st.write("Camera is not available or not accessible.")

# Chatbot Feature: Learn More About the Detected Species
st.subheader("Learn More About This Species")
if detected_species:
    st.write(get_species_info(detected_species))

# Create a species map
species_map = pd.DataFrame({
    "lat": [23.6345, -3.4653, 35.6895],  # Example latitudes
    "lon": [102.5528, 102.5041, 139.6917],  # Example longitudes
    "species": ["Tiger", "Orangutan", "Panda"]
})
st.map(species_map)
