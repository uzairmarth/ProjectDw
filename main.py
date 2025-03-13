import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Define the video capture object
cap = cv2.VideoCapture(0)

fig, ax = plt.subplots()
im = ax.imshow(np.zeros((224, 224, 3)))

while True:
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
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Update the plot with the frame and prediction
    ax.clear()
    im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Class: {class_name[2:].strip()}  Confidence: {confidence_score:.2f}")
    plt.axis('off')
    plt.pause(0.01)

cap.release()
plt.close()
