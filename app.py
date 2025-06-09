import streamlit as st
from utils import set_background, classifier
from PIL import Image

# Hintergrund setzen (optional)
#set_background('/Users/mac/Desktop/design3.png')

# Titel und Upload
st.title("Klassifikation von Straßenschildern")
st.header("Bitte laden Sie ein Bild hoch")

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption="Hochgeladenes Bild", width=150)

    if st.button("Vorhersage starten"):
        # Konfigurationspfade (passe sie ggf. an)
        MODEL_PATH = "/Users/mac/Desktop/Nachher/models/model_traffic_signs.pth"
        LABEL_ENCODER_PATH = "/Users/mac/Desktop/Nachher/label_encoder.pkl"

        # Vorhersage
        label = classifier(
            image=image,
            model_path=MODEL_PATH,
            label_encoder_path=LABEL_ENCODER_PATH
        )

        st.markdown(
    f"""
    <div style='
        background-color: #e6f2ff;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #99ccff;
        text-align: center;
        font-size: 1.2rem;
        color: #003366;
        margin-top: 1.5rem;
    '>
        <strong>Vorhergesagte Klasse:</strong> {label}
    </div>
    """,
    unsafe_allow_html=True
)

  
