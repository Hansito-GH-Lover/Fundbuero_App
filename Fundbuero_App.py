
import streamlit as st
import tensorflow as tf
import numpy as np
from supabase import create_client
from datetime import datetime
import uuid

# -------------------------------------------------
# Supabase Verbindung
# -------------------------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------------------------
# Modell laden
# -------------------------------------------------
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

def load_labels():
    with open(LABELS_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels()

# -------------------------------------------------
# Bildvorverarbeitung
# -------------------------------------------------
def preprocess_image(uploaded_file):
    img = tf.keras.utils.load_img(uploaded_file, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1
    return img_array

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("📦 Digitales Fundbüro")

uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png"])
beschreibung = st.text_input("Beschreibung")
fundort = st.text_input("Fundort")

if uploaded_file is not None:
    st.image(uploaded_file)

    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)

    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]
    confidence = float(np.max(prediction))

    st.success(f"Erkannt: {predicted_label} ({confidence:.2f})")

    if st.button("Fund speichern"):

        # -----------------------------------------
        # Bild in Supabase Storage speichern
        # -----------------------------------------
        file_name = f"{uuid.uuid4()}.jpg"

        supabase.storage.from_("fundbuero").upload(
            file_name,
            uploaded_file.getvalue(),
            {"content-type": "image/jpeg"}
        )

        public_url = supabase.storage.from_("fundbuero").get_public_url(file_name)

        # -----------------------------------------
        # Datenbankeintrag erstellen
        # -----------------------------------------
        supabase.table("fundstuecke").insert({
            "kategorie": predicted_label,
            "beschreibung": beschreibung,
            "fundort": fundort,
            "bild_url": public_url,
            "status": "Offen"
        }).execute()

        st.success("Fund gespeichert!")

# -------------------------------------------------
# Anzeige aller Funde
# -------------------------------------------------
st.subheader("📋 Aktuelle Fundstücke")

response = supabase.table("fundstuecke").select("*").execute()
items = response.data

for item in items:
    st.image(item["bild_url"], width=200)
    st.write(f"**Kategorie:** {item['kategorie']}")
    st.write(f"Beschreibung: {item['beschreibung']}")
    st.write(f"Fundort: {item['fundort']}")
    st.write(f"Status: {item['status']}")
    st.write("---")
