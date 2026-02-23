import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from datetime import datetime

# -------------------------------------------------
# Konfiguration
# -------------------------------------------------
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
DATA_FILE = "fundbuero.csv"

# -------------------------------------------------
# Modell laden
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# Labels laden
def load_labels():
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

labels = load_labels()

# -------------------------------------------------
# Bildvorverarbeitung (Teachable Machine Standard)
# -------------------------------------------------
def preprocess_image(uploaded_file):
    img = tf.keras.utils.load_img(uploaded_file, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1
    return img_array

# -------------------------------------------------
# CSV Datei vorbereiten
# -------------------------------------------------
if not os.path.exists(DATA_FILE):
    df = pd.DataFrame(columns=["Datum", "Kategorie", "Beschreibung", "Status"])
    df.to_csv(DATA_FILE, index=False)

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("📦 Digitales Fundbüro")

uploaded_file = st.file_uploader("Foto des Fundstücks hochladen", type=["jpg", "jpeg", "png"])

beschreibung = st.text_input("Beschreibung (optional)")
fundort = st.text_input("Fundort (optional)")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Hochgeladenes Bild")

    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)
    
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]
    confidence = float(np.max(prediction))

    st.subheader("🔎 Erkannte Kategorie:")
    st.success(f"{predicted_label} ({confidence:.2f})")

    if st.button("Fund speichern"):
        new_entry = {
            "Datum": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Kategorie": predicted_label,
            "Beschreibung": beschreibung,
            "Status": "Offen"
        }

        df = pd.read_csv(DATA_FILE)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)

        st.success("Fund erfolgreich gespeichert!")

# -------------------------------------------------
# Bestehende Funde anzeigen
# -------------------------------------------------
st.subheader("📋 Aktuelle Fundstücke")

df = pd.read_csv(DATA_FILE)

search = st.text_input("Suche nach Kategorie oder Beschreibung")

if search:
    df = df[df.apply(lambda row: search.lower() in str(row).lower(), axis=1)]

st.dataframe(df)
