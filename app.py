import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Load Model
model = tf.keras.models.load_model("mnist_ann_model.h5")

st.title("✍️ Draw a Digit (ANN MNIST)")

st.write("Draw a digit below 👇")

# Canvas
canvas_result = st_canvas(
    fill_color="black",  # background
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict Button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        
        img = canvas_result.image_data
        
        # Convert to grayscale
        img = Image.fromarray((img[:, :, 0]).astype('uint8'))
        
        # Resize to 28x28
        img = img.resize((28, 28))
        
        # Convert to numpy
        img_array = np.array(img)
        
        # Normalize
        img_array = img_array / 255.0
        
        # Flatten
        img_array = img_array.reshape(1, 28*28)
        
        # Prediction
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        
        st.success(f"Predicted Digit: {digit}")
