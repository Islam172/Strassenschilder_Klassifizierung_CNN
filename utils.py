#Author: Islam Elmaaroufi


import base64
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import pickle

import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)



# === Konfiguration ===
IMAGE_SIZE = (224, 224)

# === Klassifikationsfunktion ===
def classifier(image: Image.Image, model_path: str, label_encoder_path: str) -> str:
    """
    Nimmt ein PIL-Bild und Pfade zum Modell & LabelEncoder,
    und gibt die vorhergesagte Klasse als String zurück.
    """
    # Transform wie beim Training
    transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Bild vorbereiten
    input_tensor = transform(image).unsqueeze(0)  # (1, C, H, W)

    # LabelEncoder laden
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Modell vorbereiten
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(label_encoder.classes_))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Vorhersage
    with torch.no_grad():
        output = model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    return predicted_label
