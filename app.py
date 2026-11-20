import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---- CHANGE THIS to your Hugging Face model repo ----
MODEL_ID = "YOUR_HF_USERNAME/bean-leaf-vit"

st.set_page_config(page_title="Bean Leaf Disease Detector", page_icon="🌿", layout="centered")

ADVICE = {
    "healthy": "The leaf looks healthy. Keep monitoring regularly.",
    "angular_leaf_spot": "Angular leaf spot detected. Remove infected leaves, avoid overhead watering, rotate crops.",
    "bean_rust": "Bean rust detected. Remove infected plants, improve air circulation, consider a suitable fungicide.",
}


@st.cache_resource(show_spinner="Loading model from Hugging Face...")
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def predict(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}


st.title("🌿 Bean Leaf Disease Detector")
st.write(
    "Upload a photo of a bean leaf and the fine-tuned Vision Transformer will "
    "classify it as **healthy**, **angular leaf spot**, or **bean rust**."
)

processor, model = load_model()

source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"]) if source == "Upload" \
    else st.camera_input("Take a photo")

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)
    with st.spinner("Analyzing..."):
        scores = predict(image, processor, model)
    top = max(scores, key=scores.get)
    st.subheader(f"Prediction: {top.replace('_', ' ').title()}")
    st.metric("Confidence", f"{scores[top] * 100:.1f}%")
    st.info(ADVICE.get(top, ""))
    df = pd.DataFrame({"class": list(scores.keys()), "probability": list(scores.values())}).set_index("class")
    st.bar_chart(df)

st.caption("Model: fine-tuned google/vit-base-patch16-224-in21k on the Beans dataset.")
