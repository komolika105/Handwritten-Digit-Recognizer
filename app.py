import streamlit as st
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load model
model = load_model("digit_model.h5")

st.title("Real-Time Handwritten Digit Recognizer")

st.write("Draw a digit below")

# Create canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:

    # Get image data
    img = canvas_result.image_data

    # Convert to grayscale
    gray = cv2.cvtColor(
        img.astype('uint8'),
        cv2.COLOR_RGBA2GRAY
    )

    # Resize
    resized = cv2.resize(gray, (28, 28))

    # Normalize
    resized = resized / 255.0

    # Reshape
    reshaped = resized.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(reshaped)

    digit = np.argmax(prediction)

    confidence = np.max(prediction)

    st.subheader(f"Prediction: {digit}")

    st.write(f"Confidence: {confidence:.2f}")