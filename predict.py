import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("digit_model.h5")

# Load image
image = cv2.imread("sample_images/digit.png", cv2.IMREAD_GRAYSCALE)

# Resize image
image = cv2.resize(image, (28, 28))

# Invert colors
image = 255 - image

# Normalize
image = image / 255.0

# Reshape
image = image.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(image)

digit = np.argmax(prediction)

print("Predicted Digit:", digit)