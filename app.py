import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
import joblib
from streamlit_folium import st_folium
from utils.mapping import culture_mapping

# -------------------------
# ⚙️ Configuration de la page
# -------------------------
st.set_page_config(page_title="🌾 Recommandation de Culture", page_icon="🌿", layout="wide")

# -------------------------
# 📥 Chargement des données
# -------------------------
@st.cache_data
def load_data():
    try:
        gdf = gpd.read_file(r'C:\Users\moudi\Desktop\Recommandation-de-culture\rpg_sample.geojson')
        gdf["CULTURE_NOM"] = gdf["CODE_CULTU"].map(culture_mapping)
        return gdf
    except Exception as e:
        st.error(f"Erreur de chargement des données : {e}")
        return None

# -------------------------
# 📦 Chargement du modèle et de l'encodeur
# -------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model/reco_model.pkl")
        encoder = joblib.load("model/label_encoder.pkl")
        return model, encoder
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {e}")
        return None, None

gdf = load_data()
model, label_encoder = load_model()

# -------------------------
# 🧭 Interface utilisateur
# -------------------------
if gdf is not None and not gdf.empty and model is not None:
    st.sidebar.header("🧭 Navigation")
    selected_id = st.sidebar.selectbox("Sélectionne une parcelle :", gdf["ID_PARCEL"].unique())
    selected_parcel = gdf[gdf["ID_PARCEL"] == selected_id]

    st.title("🌿 Outil de Recommandation de Culture")
    st.markdown("Cette application propose une suggestion de culture suivante via un modèle Machine Learning entraîné sur des règles simples.")

    # -------------------------
    # 🗂️ Données parcelle
    # -------------------------
    st.subheader("📄 Informations sur la parcelle")
    st.dataframe(selected_parcel[["ID_PARCEL", "SURF_PARC", "CODE_CULTU", "CULTURE_NOM"]], use_container_width=True)

    # -------------------------
    # 🗺️ Carte interactive
    # -------------------------
    st.subheader("🗺️ Localisation de la parcelle")
    centroid = selected_parcel.geometry.centroid.iloc[0]
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14)
    folium.GeoJson(selected_parcel.geometry).add_to(m)
    st_folium(m, height=400, width=700)

    # -------------------------
    # 🔮 Recommandation via ML
    # -------------------------
    st.subheader("🔮 Recommandation de culture via modèle ML")

    try:
        code_actuel = selected_parcel["CODE_CULTU"].values[0]
        surface = selected_parcel["SURF_PARC"].values[0]

        # Encoder la culture actuelle
        encoded_input = label_encoder.transform([code_actuel])[0]
        X_input = pd.DataFrame([[encoded_input, surface]], columns=["culture_actuelle_code", "surface"])

        # Prédiction
        prediction_encoded = model.predict(X_input)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
        culture_reco = culture_mapping.get(prediction_label, prediction_label)

        st.success(f"👉 Culture recommandée par le modèle : **{culture_reco}**")

    except Exception as e:
        st.warning(f"🚫 Le modèle ne peut pas prédire pour cette culture : {e}")

else:
    st.warning("⚠️ Les données ou le modèle ne sont pas disponibles.")
