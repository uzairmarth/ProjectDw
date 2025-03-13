import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

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

# Define the video capture object
cap = cv2.VideoCapture(0)

stframe = st.empty()
detection_placeholder = st.empty()
info_placeholder = st.empty()

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

    # Get animal information
    if class_name in animal_info:
        info = animal_info[class_name]
    else:
        info = "No information available."

    # Update the frame and detection information
    stframe.image(frame, channels="BGR")
    detection_placeholder.text(f"Class: {class_name}  Confidence: {confidence_score:.2f}")
    info_placeholder.text(f"Information: {info}\n{estimated_age}")

cap.release()
