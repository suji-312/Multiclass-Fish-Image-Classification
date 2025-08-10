import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Fish Image Classification", page_icon="🐟", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("efficientnet_fish_classifier_final.h5", compile=False)

model = load_model()

class_names = [
    "fish sea_food trout",
    "fish sea_food striped_red_mullet",
    "fish sea_food shrimp",
    "fish sea_food sea_bass",
    "fish sea_food red_sea_bream",
    "fish sea_food red_mullet",
    "fish sea_food hourse_mackerel",
    "fish sea_food gilt_head_bream",
    "fish sea_food black_sea_sprat",
    "animal fish",
    "animal fish bass"
]

def preprocess_image(image):
    img = image.convert("L")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("🐟 Fish Image Classification")
st.write("Upload an image of a fish and get the predicted category with confidence scores.")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(img)
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    st.subheader(f"✅ Predicted Class: **{predicted_class}**")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")

    # Check lengths match
    if len(class_names) != len(predictions[0]):
        st.error("Mismatch between number of class names and model outputs!")
        st.stop()

    st.write("### Confidence Scores by Class:")
    # Show confidence scores as text only
    for cls, conf in zip(class_names, predictions[0] * 100):
        st.write(f"- {cls}: {conf:.2f}%")


