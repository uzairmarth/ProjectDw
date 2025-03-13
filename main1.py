import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import pandas as pd
import time
import wikipedia
import pydeck as pdk

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Dictionary with animal information
animal_info = {
    "Elephant": "Sumatran elephants are critically endangered due to habitat loss and poaching.",
    "Tiger": "Sunda tigers are a subspecies of tigers found primarily in the Sunda Islands of Indonesia.",
    "Rhino": "Black rhinos are critically endangered and are native to eastern and southern Africa."
}

# Function to get species info from Wikipedia
def get_species_info(species):
    try:
        summary = wikipedia.summary(species, sentences=2)
    except:
        summary = "No additional information available."
    return summary

# Initialize Streamlit app
st.title("Animal Detection Dashboard")
st.sidebar.header("Wildlife Detection Settings")
upload_option = st.sidebar.radio("Choose Input Source:", ("Upload Image(s)", "Use Webcam"))

# Store results in a DataFrame
results_df = pd.DataFrame(columns=["Timestamp", "Detected Species", "Confidence"])
confidence_scores = []
time_series = []
start_time = time.time()

if upload_option == "Upload Image(s)":
    uploaded_files = st.file_uploader("Upload one or more images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
            # Preprocess the image
            image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array
            # Predict using the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index].strip().split(" ")[1]
            confidence_score = prediction[0][index]
            confidence_scores.append(confidence_score)
            time_series.append(time.time() - start_time)
            # Get animal information
            info = animal_info.get(class_name, "No information available.")
            additional_info = get_species_info(class_name)
            # Store detection results
            new_entry = pd.DataFrame([[time.strftime('%Y-%m-%d %H:%M:%S'), class_name, confidence_score]],
                                      columns=results_df.columns)
            results_df = pd.concat([results_df, new_entry], ignore_index=True)
            # Display results
            st.subheader(f"Detected: {class_name} ({confidence_score:.2f}%)")
            st.write(f"Information: {info}\n{additional_info}")

elif upload_option == "Use Webcam":
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
        confidence_scores.append(confidence_score)
        time_series.append(time.time() - start_time)
        # Get animal information
        info = animal_info.get(class_name, "No information available.")
        additional_info = get_species_info(class_name)
        # Store detection results
        new_entry = pd.DataFrame([[time.strftime('%Y-%m-%d %H:%M:%S'), class_name, confidence_score]],
                                  columns=results_df.columns)
        results_df = pd.concat([results_df, new_entry], ignore_index=True)
        # Update the frame and detection information
        stframe.image(frame, channels="BGR")
        detection_placeholder.text(f"Class: {class_name}  Confidence: {confidence_score:.2f}")
        info_placeholder.text(f"Information: {info}\n{additional_info}")
    cap.release()

    # Plot the real-time confidence scores
    st.subheader("Real-Time Confidence Score")
    fig, ax = plt.subplots()
    ax.plot(time_series, confidence_scores, label="Confidence %")
    ax.set_xlabel("Time")
    ax.set_ylabel("Confidence")
    ax.legend()
    st.pyplot(fig)

# Display top detected species leaderboard
top_detections = results_df["Detected Species"].value_counts().head(5)
st.subheader("Top Detected Species")
st.write(top_detections)

# Save results to CSV
results_df.to_csv("detection_results.csv", index=False)

# Create a species map
species_map = pd.DataFrame({
    "lat": [23.6345, -3.4653, 35.6895],  # Example latitudes
    "lon": [102.5528, 102.5041, 139.6917],  # Example longitudes
    "species": ["Tiger", "Orangutan", "Panda"]
})
st.map(species_map)
