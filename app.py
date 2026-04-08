import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import time


st.set_page_config(page_title="AI Animal Healthcare System", layout="wide")


@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load('outbreak_model.pkl')
        le_country = joblib.load('le_country.pkl')
        le_species = joblib.load('le_species.pkl')
        le_disease = joblib.load('le_disease.pkl')
        return model, le_country, le_species, le_disease, True
    except:
        return None, None, None, None, False

model, le_country, le_species, le_disease, ml_ready = load_ml_assets()


if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'auth_mode' not in st.session_state:
    st.session_state['auth_mode'] = 'Login'


if not st.session_state['logged_in']:
    bg_style = """
    <style>
    .stApp {
        background-color: #f0f9ff;
        background-image:  
            radial-gradient(#3b82f6 0.8px, transparent 0.8px), 
            linear-gradient(rgba(59, 130, 246, 0.05) 2px, transparent 2px), 
            linear-gradient(90deg, rgba(59, 130, 246, 0.05) 2px, transparent 2px);
        background-size: 30px 30px, 60px 60px, 60px 60px;
        background-position: 0 0, 30px 30px, 30px 30px;
    }
    </style>
    """
else:
    bg_style = """
    <style>
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(216, 241, 223, 0.6) 0%, rgba(248, 250, 252, 1) 90%);
        background-color: #f0fdf4;
    }
    </style>
    """

st.markdown(bg_style, unsafe_allow_html=True)


LANG_DICT = {
    "English": {
        "title": "🐾 AI-Based Animal Disease Prediction System",
        "subtitle": "Early identification of serious health conditions using Machine Learning.",
        "animal_info": "📋 Animal Information", "upload": "📸 Upload Image",
        "results": "🔍 Analysis Results", "analyze": "Analyze Disease",
        "validate": "✅ Image Uploaded Successfully",
        "high_sev": "🚨 High Severity: Immediate Veterinary Consultation is recommended.",
        "low_sev": "⚠️ Monitor symptoms and consult a professional if condition persists.",
        "welcome": "Welcome to the Healthcare Portal", "logout_btn": "Logout", "records_btn": "View Records",
        "help_msg": "Note: For a single sick animal, keep Observed Cases as 1 and Deaths as 0.",
        "suggestion_title": "💡 AI Suggestions & Next Steps"
    },
    "Hindi": {
        "title": "🐾 AI-आधारित पशु रोग भविष्यवाणी प्रणाली",
        "subtitle": "मशीन लर्निंग का उपयोग करके गंभीर स्वास्थ्य स्थितियों की प्रारंभिक पहचान।",
        "animal_info": "📋 पशु जानकारी", "upload": "📸 इमेज अपलोड करें",
        "results": "🔍 विश्लेषण परिणाम", "analyze": "रोग का विश्लेषण करें",
        "validate": "✅ छवि सफलतापूर्वक अपलोड की गई",
        "high_sev": "🚨 उच्च गंभीरता: तत्काल पशु चिकित्सक से परामर्श की सिफारिश की जाती है।",
        "low_sev": "⚠️ लक्षणों की निगरानी करें और स्थिति बनी रहने पर किसी पेशेवर से सलाह लें।",
        "welcome": "हेल्थकेयर पोर्टल में स्वागत है", "logout_btn": "लॉगआउट", "records_btn": "रिकॉर्ड देखें",
        "help_msg": "नोट: एक बीमार जानवर के लिए, 'Observed Cases' को 1 और 'Deaths' को 0 रखें।",
        "suggestion_title": "💡 AI सुझाव और अगले कदम"
    },
    "Tamil": {
        "title": "🐾 AI-அடிப்படையிலான விலங்கு நோய் கணிப்பு அமைப்பு",
        "subtitle": "இயந்திர கற்றலைப் பயன்படுத்தி தீவிர சுகாதார நிலைகளை முன்கூட்டியே கண்டறிதல்.",
        "animal_info": "📋 விலங்கு தகவல்", "upload": "📸 படத்தை பதிவேற்றவும்",
        "results": "🔍 பகுப்பாய்வு முடிவுகள்", "analyze": "நோயைப் பகுப்பாய்வு செய்யுங்கள்",
        "validate": "✅ படம் வெற்றிகரமாக பதிவேற்றப்பட்டது",
        "high_sev": "🚨 அதிக தீவிரம்: உடனடியாக கால்நடை மருத்துவரை அணுக பரிந்துரைக்கப்படுகிறது.",
        "low_sev": "⚠️ அறிகுறிகளைக் கண்காணித்து, நிலைமை நீடித்தால் நிபுணரை அணுகவும்.",
        "welcome": "சுகாதார போர்ட்டலுக்கு உங்களை வரவேற்கிறோம்", "logout_btn": "வெளியேறு", "records_btn": "பதிவுகளைப் பார்க்கவும்",
        "help_msg": "குறிப்பு: ஒரு விலங்குக்கு மட்டும், 'Observed Cases' 1 மற்றும் 'Deaths' 0 என வைக்கவும்.",
        "suggestion_title": "💡 AI பரிந்துரைகள் மற்றும் அடுத்த படிகள்"
    }
}

def logout():
    st.session_state['logged_in'] = False
    st.rerun()


def predict_disease_with_model(country, species, is_wild, cases, deaths):
    if not ml_ready:
        return "Model Offline", 0.0, "The ML model is currently unavailable."
    
    try:
        c_enc = le_country.transform([country])[0]
        s_enc = le_species.transform([species])[0]
        wild_val, domestic_val, aquatic_val = (1, 0, 0) if is_wild else (0, 1, 0)
        med_lat, med_lon, med_susceptible = 20.59, 78.96, 500
        
        features = np.array([[c_enc, s_enc, wild_val, domestic_val, aquatic_val, med_lat, med_lon, med_susceptible, cases, deaths]])
        
        pred_label = model.predict(features)[0]
        probs = model.predict_proba(features)
        confidence = np.max(probs) * 100
        disease_name = le_disease.inverse_transform([pred_label])[0]
        

        # High Risk if deaths > 0 or cases > 5 or confidence is high for known dangerous diseases
        is_risk = deaths > 0 or cases > 5 or confidence > 85
        status = "**RISK ALERT**" if is_risk else "**NORMAL/STABLE**"
        
        suggestions = {
            "Foot and Mouth Disease": f"{status}: Isolate the animal. Clean water troughs. You must visit a veterinary consult for specialized care.",
            "Lumpy Skin Disease": f"{status}: Apply antiseptic to lesions. Use insect repellent. Please visit a veterinary consult immediately.",
            "Avian Influenza": f"{status}: High transmission risk. Cull infected birds. You must visit a veterinary consult to prevent further outbreak.",
            "Rabies": f"{status}: CRITICAL. Do NOT approach. Secure the animal and visit a veterinary consult immediately.",
            "Anthrax": f"{status}: CRITICAL. Do NOT open carcass. Avoid contact and visit a veterinary consult for preventative measures."
        }
        

        advice = suggestions.get(disease_name, f"{status}: Ensure isolation and clean water. Please visit a veterinary consult for a professional diagnosis.")
        
        return disease_name, confidence, advice
    except Exception as e:
        return "Error", 0.0, f"Analysis failed: {str(e)}"

if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.session_state['auth_mode'] == 'Login':
            st.title("🔐 Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login", use_container_width=True):
                st.session_state['logged_in'] = True
                st.rerun()
            if st.button("Sign Up", use_container_width=True):
                st.session_state['auth_mode'] = 'Signup'
                st.rerun()
        else:
            st.title("📝 Register")
            st.text_input("Name")
            st.text_input("Email")
            st.text_input("Password", type="password")
            if st.button("Register", use_container_width=True):
                st.session_state['auth_mode'] = 'Login'
                st.rerun()

else:
    st.sidebar.title("👤 Menu")
    if st.sidebar.button("Logout"): logout()
    lang = st.sidebar.selectbox("🌐 Language", ["English", "Hindi", "Tamil"])
    texts = LANG_DICT[lang]

    st.title(texts["title"])
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.header(texts["animal_info"])
        if ml_ready:
            species_list, country_list = list(le_species.classes_), list(le_country.classes_)
        else:
            species_list, country_list = ["Generic"], ["Global"]
        
        sel_species = st.selectbox("Species", species_list)
        sel_country = st.selectbox("Country", country_list)
        is_wild = st.checkbox("Wild Animal")
        num_cases = st.number_input("Cases", min_value=1, value=1)
        num_deaths = st.number_input("Deaths", min_value=0, value=0)
        
        st.header(texts["upload"])
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="uploader")

    with col2:
        st.header(texts["results"])
        if uploaded_file:
            st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)
            st.success(texts["validate"])
            
            if st.button(texts["analyze"], type="primary", use_container_width=True):
                with st.spinner('Analyzing...'):
                    time.sleep(1)
                    disease, score, suggestion = predict_disease_with_model(sel_country, sel_species, is_wild, num_cases, num_deaths)
                    
                    st.success(f"### Accurate Diagnosis: {disease}")5
                    st.metric("Model Confidence", f"{score:.2f}%")
                    st.info(f"**{texts['suggestion_title']}**\n\n{suggestion}")
                    
                    if score > 80 or num_deaths > 0: 
                        st.error(texts["high_sev"])
                    else: 
                        st.warning(texts["low_sev"])
        else:
            st.info("Please upload an image to proceed with the analysis results.")