import streamlit as st
from supabase import create_client
import uuid
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import requests  # nur noch für Supabase, nicht mehr für Klassifikation

# =====================================
# SUPABASE VERBINDUNG
# =====================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "fundbuero"

# =====================================
# MODELL LADEN – ein Mal beim App-Start
# =====================================

MODEL_URL = "https://raw.githubusercontent.com/Hansito-GH-Lover/Fundbuero_App/main/keras_model.h5"

@st.cache_resource(show_spinner="Lade Klassifikations-Modell ... (einmalig)")
def load_classification_model():
    try:
        # Direkt von GitHub laden
        model = load_model(MODEL_URL)
        st.success("Modell erfolgreich geladen!")
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {str(e)}\n\n→ Prüfe, ob die URL korrekt ist und die Datei öffentlich zugänglich.")
        return None

model = load_classification_model()

# Deine Klassen – Reihenfolge muss exakt mit dem Training übereinstimmen!
CLASSES = [
    "Brotdose",
    "Hose",
    "T-Shirt",
    "Pullover",
    "Trinkflasche"
]

def classify_image(image_file):
    if model is None:
        return "Modell nicht verfügbar"

    try:
        # Bild laden und vorbereiten
        img = Image.open(io.BytesIO(image_file.getvalue())).convert("RGB")
        img = img.resize((224, 224))           # ← Standard für viele TM-Modelle; ggf. anpassen!
        img_array = np.array(img) / 255.0       # Normalisierung [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen

        # Vorhersage
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])

        class_name = CLASSES[predicted_idx]
        return f"{class_name} ({confidence:.1%})"

    except Exception as e:
        st.warning(f"Fehler bei der Bild-Klassifikation: {str(e)}")
        return "Unbekannt"

# =====================================
# STREAMLIT UI
# =====================================

st.title("🏫 Schul-Fundbüro")

st.write("Lade ein gefundenes Objekt hoch.")

uploaded_file = st.file_uploader(
    "Foto des Fundstücks",
    type=["jpg", "jpeg", "png"]
)

beschreibung = st.text_input("Beschreibung")
fundort = st.text_input("Fundort")

# =====================================
# UPLOAD BUTTON
# =====================================

if st.button("Fundstück speichern"):

    if uploaded_file is None:
        st.error("Bitte ein Bild hochladen.")
    elif not uploaded_file.type.startswith("image/"):
        st.error("Nur Bilddateien erlaubt (jpg, jpeg, png).")
    else:
        # Kategorie über lokales Modell bestimmen
        with st.spinner("Klassifiziere Bild ..."):
            kategorie = classify_image(uploaded_file)

        # Dateiname erzeugen
        filename = f"{uuid.uuid4()}.jpg"

        # Bild zu Supabase hochladen
        uploaded_file.seek(0)  # Wichtig: Zeiger zurücksetzen!
        supabase.storage.from_(BUCKET_NAME).upload(
            filename,
            uploaded_file.getvalue()
        )

        # Öffentliche URL
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{filename}"

        # In Datenbank speichern
        data = {
            "kategorie": kategorie,
            "beschreibung": beschreibung,
            "fundort": fundort,
            "bild_url": image_url,
            "status": "Offen"
        }

        supabase.table("fundbuero").insert(data).execute()

        st.success("Fundstück gespeichert!")

# =====================================
# FUNDSTÜCKE ANZEIGEN
# =====================================

st.header("Gefundene Gegenstände")

response = supabase.table("fundbuero").select("*").order("created_at", desc=True).execute()

for item in response.data:
    st.image(item["bild_url"], width=200)
    st.write("**Kategorie:**", item["kategorie"])
    st.write("**Beschreibung:**", item["beschreibung"])
    st.write("**Fundort:**", item["fundort"])
    st.write("**Status:**", item["status"])
    st.markdown("---")
