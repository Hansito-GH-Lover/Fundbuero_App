import streamlit as st
from supabase import create_client
import uuid
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# =============================================
# SUPABASE KONFIGURATION
# =============================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "fundbuero"

# =============================================
# KLASSIFIKATIONSMODELL LADEN (cached!)
# =============================================

@st.cache_resource(show_spinner="Lade Klassifizierungsmodell …")
def load_classification_model():
    try:
        # Datei muss im Repository-Root liegen
        model = load_model("keras_model.h5")
        st.success("Modell erfolgreich geladen")
        return model
    except Exception as e:
        st.error(f"Modell konnte nicht geladen werden:\n{str(e)}\n\n→ Ist keras_model.h5 im Repository-Root?")
        st.info("Mögliche Ursachen:\n1. Datei fehlt im Git-Repository\n2. Falscher Dateiname\n3. Datei ist beschädigt oder falsches Format")
        return None

model = load_classification_model()

# WICHTIG: Reihenfolge MUSS mit deinem Teachable Machine / Keras Training übereinstimmen!
CLASSES = [
    "Brotdose",     # Index 0
    "Hose",         # 1
    "T-Shirt",      # 2
    "Pullover",     # 3
    "Trinkflasche"  # 4
]

def classify_image(image_file):
    if model is None:
        return "Modell nicht verfügbar"

    try:
        # Bild vorbereiten
        img = Image.open(io.BytesIO(image_file.getvalue())).convert("RGB")
        img = img.resize((224, 224))                  # ← Standard bei Teachable Machine – ggf. anpassen!
        img_array = np.array(img) / 255.0             # Normalisierung
        img_array = np.expand_dims(img_array, axis=0) # Batch-Dimension

        # Vorhersage
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])

        if confidence < 0.35:  # ← optional – Schwelle gegen Unsicherheit
            return "Unbekannt (zu unsicher)"

        return f"{CLASSES[predicted_idx]} ({confidence:.0%})"

    except Exception as e:
        st.warning(f"Klassifikation fehlgeschlagen: {str(e)}")
        return "Fehler"

# =============================================
# STREAMLIT OBERFLÄCHE
# =============================================

st.title("🏫 Schul-Fundbüro")
st.write("Funde hochladen – die App erkennt automatisch die Kategorie")

with st.sidebar:
    st.markdown("### Hinweise")
    st.info("Erlaubte Dateien: jpg, jpeg, png\n\nKategorien: Brotdose, Hose, T-Shirt, Pullover, Trinkflasche")

uploaded_file = st.file_uploader(
    "Foto des Fundstücks",
    type=["jpg", "jpeg", "png"],
    help="Mache ein Foto vom Fundstück"
)

beschreibung = st.text_input("Beschreibung (optional)")
fundort = st.text_input("Fundort (z. B. Pausenhof, Klasse 8b, …)")

if st.button("Fundstück melden", type="primary", use_container_width=True):

    if not uploaded_file:
        st.error("Bitte ein Foto hochladen")
    else:
        with st.spinner("Bild wird analysiert und gespeichert …"):
            # Klassifizieren
            kategorie = classify_image(uploaded_file)

            # Dateiname eindeutig machen
            filename = f"{uuid.uuid4()}.jpg"

            # Bild hochladen (Zeiger zurücksetzen!)
            uploaded_file.seek(0)
            supabase.storage.from_(BUCKET_NAME).upload(
                filename,
                uploaded_file.getvalue(),
                file_options={"content-type": "image/jpeg"}
            )

            # Öffentliche URL erstellen
            image_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{filename}"

            # Datensatz speichern
            data = {
                "kategorie": kategorie,
                "beschreibung": beschreibung.strip() or None,
                "fundort": fundort.strip() or None,
                "bild_url": image_url,
                "status": "Offen"
            }

            try:
                supabase.table("fundbuero").insert(data).execute()
                st.success("✓ Fundstück erfolgreich gemeldet!")
            except Exception as e:
                st.error(f"Datenbank-Fehler: {str(e)}")

# =============================================
# ANZEIGE DER FUNDSACHE
# =============================================

st.header("Gefundene Gegenstände")

try:
    response = supabase.table("fundbuero") \
        .select("*") \
        .order("created_at", desc=True) \
        .execute()

    if not response.data:
        st.info("Noch keine Fundstücke gemeldet.")
    else:
        for item in response.data:
            cols = st.columns([1, 3])
            with cols[0]:
                st.image(item["bild_url"], width=140)
            with cols[1]:
                st.markdown(f"**{item['kategorie']}**")
                if item["beschreibung"]:
                    st.write(item["beschreibung"])
                st.caption(f"Fundort: {item['fundort'] or '—'}")
                st.caption(f"Status: {item['status']}")
            st.markdown("---")

except Exception as e:
    st.error(f"Fehler beim Laden der Fundstücke: {str(e)}")
