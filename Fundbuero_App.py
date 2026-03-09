import streamlit as st
from supabase import create_client
import requests
from datetime import datetime
import uuid

# ==========================
# SUPABASE VERBINDUNG
# ==========================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "fundbuero"

# ==========================
# TEACHABLE MACHINE
# ==========================

MODEL_URL = st.secrets["MODEL_URL"]

def classify_image(image_file):

    try:
        response = requests.post(
            MODEL_URL,
            files={"image": image_file}
        )

        result = response.json()

        return result["class"]

    except:
        return "Unbekannt"


# ==========================
# STREAMLIT UI
# ==========================

st.title("🏫 Schul-Fundbüro")

st.write("Lade ein gefundenes Objekt hoch.")

uploaded_file = st.file_uploader(
    "Foto des Fundstücks",
    type=["jpg", "jpeg", "png"]
)

beschreibung = st.text_input("Beschreibung")
fundort = st.text_input("Fundort")

# ==========================
# UPLOAD BUTTON
# ==========================

if st.button("Fundstück speichern"):

    if uploaded_file is None:
        st.error("Bitte ein Bild hochladen.")
    else:

        # Kategorie über KI bestimmen
        kategorie = classify_image(uploaded_file)

        # Dateiname erzeugen
        filename = f"{uuid.uuid4()}.jpg"

        # Bild zu Supabase hochladen
        supabase.storage.from_(BUCKET_NAME).upload(
            filename,
            uploaded_file.getvalue()
        )

        # Öffentliche URL erzeugen
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

# ==========================
# FUNDSTÜCKE ANZEIGEN
# ==========================

st.header("Gefundene Gegenstände")

response = supabase.table("fundbuero").select("*").order("created_at", desc=True).execute()

for item in response.data:

    st.image(item["bild_url"], width=200)

    st.write("Kategorie:", item["kategorie"])
    st.write("Beschreibung:", item["beschreibung"])
    st.write("Fundort:", item["fundort"])
    st.write("Status:", item["status"])

    st.markdown("---")
