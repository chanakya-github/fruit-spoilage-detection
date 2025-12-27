import streamlit as st
import cv2
import numpy as np
import joblib

def check_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    return brightness

# Load trained model
model = joblib.load("models/fruit_spoilage_rf_model.pkl")

st.title("Food Spoilage Detection")
st.write("Upload an image to classify as Fresh or Rotten")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    brightness = check_image_quality(image)

    if brightness < 50:
        st.warning("Image is too dark. Please capture the image under better lighting.")
    elif brightness > 200:
        st.warning("Image is too bright. Please reduce lighting or avoid glare.")
    else:
        st.success("Image quality is good.")

    # Preprocess
    image_resized = cv2.resize(image, (100, 100))
    mean_rgb = image_resized.mean(axis=(0,1)).reshape(1, -1)

    # Get prediction probabilities

    probs = model.predict_proba(mean_rgb)[0]

    fresh_prob = probs[0] * 100
    rotten_prob = probs[1] * 100

    # Determine freshness stage
    if rotten_prob < 30:
        stage = "Fresh"
        st.success("Freshness Stage: Fresh 🟢")
    elif 30 <= rotten_prob <= 70:
        stage = "Semi-spoiled"
        st.warning("Freshness Stage: Semi-spoiled 🟡")
    else:
        stage = "Spoiled"
        st.error("Freshness Stage: Spoiled 🔴")

    # Show confidence
    st.info(f"Confidence (Rotten): {rotten_prob:.2f}%")


