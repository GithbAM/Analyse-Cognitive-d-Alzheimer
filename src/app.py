#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


# In[9]:


# Chargement du modèle

@st.cache_resource
def load_pipeline():
    src_dir = Path(__file__).resolve().parent
    model_path = src_dir.parent / "models" / "alzheimer_pipeline_trained.pkl"
    return joblib.load(model_path)

pipeline = load_pipeline()


# In[10]:


# Prédictions

def predict_alzheimer_risk(inputs: dict):
    """Prédit le risque d'Alzheimer à partir du modèle"""

    df = pd.DataFrame([inputs])
    pred, proba = pipeline.predict_new_data(df)
    probability = proba[0, 1]

    if probability > 0.7:
        risk_level = 'Élevé'
        recommendation = 'Bilan neuropsychologique approfondi recommandé dans les 15 jours'
        color = "red"

    elif probability > 0.4:
        risk_level = 'Modéré'
        recommendation = 'Suivi rapproché et réévaluation dans 3 mois'
        color = 'orange'

    else:
        risk_level = 'Faible'
        recommendation = 'Suivi de routine, réévaluation annuelle'
        color = 'green'

    return {'probability': float(probability),
           'risk_level': risk_level,
           'recommendation': recommendation,
           'color': color}


# In[14]:


# Configuration de la page

st.set_page_config(page_title='Clinical Decision Support',
                  layout='wide')
st.title("Clinical Decision Support System")
st.subheader("Aide au diagnostic précoce de la maladie d'Alzheimer")

st.sidebar.header("Données Patient")

#Évaluations Cognitives
st.sidebar.markdown("Évaluations Cognitives")
mmse = st.sidebar.slider("Score MMSE (0-30)", 0, 30, 20)
functional_assessment = st.sidebar.slider("Score Fonctionnel (0-10)", 0.0, 10.0, 7.5, 0.1)
adl = st.sidebar.slider("Score ADL (0-10)", 0.0, 10.0, 8.2, 0.1)

#Symptomes Cognitifs
st.sidebar.markdown("Symptômes Cognitifs")
memory_complaints = st.sidebar.checkbox("Troubles mémoriels")
behavioral_problems = st.sidebar.checkbox("Troubles comportementaux")
confusion = st.sidebar.checkbox("Confusion")
disorientation = st.sidebar.checkbox("Désorientation") 
personality_changes = st.sidebar.checkbox("Changements de personnalité")
difficulty_completing_tasks = st.sidebar.checkbox("Difficultés à effectuer des tâches complexes")
forgetfulness = st.sidebar.checkbox("Oublis fréquents")

#Données Démographiques
st.sidebar.markdown("Données Démographiques")
age = st.sidebar.slider("Âge", 50, 100, 65)
gender = st.sidebar.selectbox("Genre", options=[0, 1], index=1, help="0=Femme, 1=Homme")
ethnicity = st.sidebar.selectbox("Origine ethnique", options=[0, 1, 2, 3], index=0, help="Origine ethnique (0-3)")
education_level = st.sidebar.selectbox("Niveau d'éducation", options=[0, 1, 2, 3, 4], index=2, help="Niveau d'éducation (0-4)")

# Paramètres Physiologiques
st.sidebar.markdown("Paramètres Physiologiques")
bmi = st.sidebar.slider("IMC (15-40)", 15.0, 40.0, 25.0, 0.1)
systolic_bp = st.sidebar.slider("TA Systolique (90-200)", 90, 200, 130)
diastolic_bp = st.sidebar.slider("TA Diastolique (50-120)", 50, 120, 80)
cholesterol_total = st.sidebar.slider("Cholestérol total (100-400)", 100, 400, 200)
cholesterol_ldl = st.sidebar.slider("LDL (50-250)", 50, 250, 120)
cholesterol_hdl = st.sidebar.slider("HDL (20-100)", 20, 100, 50)
cholesterol_triglycerides = st.sidebar.slider("Triglycérides (50-400)")

# Mode de Vie
st.sidebar.markdown("Mode de Vie")
diet_quality = st.sidebar.slider("Qualité de l'alimentation (0-10)", 0, 10, 7)
physical_activity = st.sidebar.slider("Activité physique (h/semaine, 0-10)", 0, 10, 4)
alcohol_consumption = st.sidebar.slider("Consommation d'alcool (verres/semaine, 0-10)", 0, 10, 2)
sleep_quality = st.sidebar.slider("Qualité du sommeil (0-10)", 0, 10, 7)
smoking = st.sidebar.checkbox("Tabagisme actuel")

# Antécédents Médicaux
st.sidebar.markdown("Antécédents Médicaux")
family_history_alzheimers = st.sidebar.checkbox("Antécédents familiaux d'Alzheimer")
cardiovascular_disease = st.sidebar.checkbox("Maladie cardiovasculaire")
diabetes = st.sidebar.checkbox("Diabète")
depression = st.sidebar.checkbox("Dépression")
head_injury = st.sidebar.checkbox("Traumatisme crânien")
hypertension = st.sidebar.checkbox("Hypertension artérielle")

# Bouton
analyze_button = st.sidebar.button("Analyser le Risque", type="primary")


# In[15]:


# Layout

col1, col2 = st.columns([2, 1])

if analyze_button:
    inputs = {
        'Age': age,
        'Gender': gender,
        'Ethnicity': ethnicity,
        'EducationLevel': education_level,
        'BMI': bmi,
        'Smoking': int(smoking),
        'AlcoholConsumption': alcohol_consumption,
        'PhysicalActivity': physical_activity,
        'DietQuality': diet_quality,
        'SleepQuality': sleep_quality,
        'FamilyHistoryAlzheimers': int(family_history_alzheimers),
        'CardiovascularDisease': int(cardiovascular_disease),
        'Diabetes': int(diabetes),
        'Depression': int(depression),
        'HeadInjury': int(head_injury),
        'Hypertension': int(hypertension),
        'SystolicBP': systolic_bp,
        'DiastolicBP': diastolic_bp,
        'CholesterolTotal': cholesterol_total,
        'CholesterolLDL': cholesterol_ldl,
        'CholesterolHDL': cholesterol_hdl,
        'CholesterolTriglycerides': cholesterol_triglycerides,
        'MMSE': mmse,
        'FunctionalAssessment': functional_assessment,
        'MemoryComplaints': int(memory_complaints),
        'BehavioralProblems': int(behavioral_problems),
        'ADL': adl,
        'Confusion': int(confusion),
        'Disorientation': int(disorientation),
        'PersonalityChanges': int(personality_changes),
        'DifficultyCompletingTasks': int(difficulty_completing_tasks),
        'Forgetfulness': int(forgetfulness)
    }

    result = predict_alzheimer_risk(inputs)

    with col1:
        st.markdown("Résultat de l'Analyse")
        st.metric(
            label="Probabilité de Risque Alzheimer",
            value=f"{result['probability']:.1%}",
            delta=f"Risque {result['risk_level']}"
        )

        if result['color'] == 'red':
            st.error(result['recommendation'])
        elif result['color'] == 'orange':
            st.warning(result['recommendation'])
        else:
            st.success(result['recommendation'])

        risk_data = pd.DataFrame({
            'Niveau': ['Faible', 'Modéré', 'Élevé'],
            'Seuil': [0.4, 0.7, 1.0],
            'Couleur': ['green', 'orange', 'red']
        })

        fig = px.bar(
            risk_data,
            x='Niveau',
            y='Seuil',
            color='Couleur',
            color_discrete_map={'green': '#22c55e', 'orange': '#f97316', 'red': '#ef4444'},
            title="Échelle de Risque"
        )
        fig.add_hline(y=result['probability'], line_dash="dash", line_color="blue",
                      annotation_text=f"Patient: {result['probability']:.1%}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
            st.markdown("## Analyse des Features")
            
            # Features principales utilisées (les 17 sélectionnées)
            st.markdown("### Features Actives (17 sélectionnées)")
            key_features = [
                ("FunctionalAssessment", functional_assessment, "8-10", "Impact maximum"),
                ("MMSE", mmse, "24-30", "Cognition"),
                ("ADL", adl, "8-10", "Autonomie"),
                ("MemoryComplaints", "Oui" if memory_complaints else "Non", "-", "Plaintes"),
                ("BehavioralProblems", "Oui" if behavioral_problems else "Non", "-", "Comportement")
            ]
            
            for feature, value, normal, description in key_features:
                if feature == "MMSE" and value < 24:
                    st.markdown(f"🔴 **{feature}**: {value} (Normal: {normal}) - {description}")
                elif feature == "FunctionalAssessment" and value < 8:
                    st.markdown(f"🔴 **{feature}**: {value} (Normal: {normal}) - {description}")
                elif feature == "ADL" and value < 8:
                    st.markdown(f"🟡 **{feature}**: {value} (Normal: {normal}) - {description}")
                elif feature in ["MemoryComplaints", "BehavioralProblems"] and value == "Oui":
                    st.markdown(f"🔴 **{feature}**: {value} - {description}")
                else:
                    st.markdown(f"🟢 **{feature}**: {value} - {description}")
            
            # Facteurs de risque supplémentaires saisis
            st.markdown("### Facteurs Contextuels Saisis")
            risk_factors = []
            if family_history_alzheimers:
                risk_factors.append("Antécédents familiaux")
            if cardiovascular_disease:
                risk_factors.append("Maladie cardiovasculaire")
            if diabetes:
                risk_factors.append("Diabète")
            if depression:
                risk_factors.append("Dépression")
            if hypertension:
                risk_factors.append("Hypertension")
            if smoking:
                risk_factors.append("Tabagisme")
                
            if risk_factors:
                st.markdown("🟡 **Facteurs présents**: " + ", ".join(risk_factors))
            else:
                st.markdown("🟢 **Aucun facteur de risque majeur**")

else:
    # Écran d'accueil
    with col1:
        st.markdown("## Interface Complète - 32 Variables")
        st.markdown(f"""
        Cette version utilise **TOUTES les variables** du dataset pour 
        une évaluation exhaustive du risque Alzheimer.
        
        ** Architecture Complète:**
        - **32 variables éditables**
        - **Scaler sur 32 features**
        - **Prédiction sur 17 features**
        - **Le modèle réel**
        
        ** Variables par catégorie:**
        - **Cognitives**: MMSE, FunctionalAssessment, ADL, symptômes
        - **Démographiques**: Âge, Genre, Éducation, Origine
        - **Physiologiques**: BMI, TA, Cholestérol complet
        - **Mode de vie**: Alimentation, Sport, Sommeil, Tabac
        - **Antécédents**: Familiaux, Cardio, Diabète, etc.
        
        **Impact clinique avec modèle complet:**
        - **91.03% précision réelle**
        - **Évaluation exhaustive**
        - **Pipeline production**
        - **Interface médicale complète**
        """)
        
    with col2:
        st.markdown("## 🔬 Détails Techniques")
        st.markdown(f"""
        ### Variables Implémentées
        **Toutes les 32 features du dataset original:**
        
        ** Cognitives (7)**: MMSE, FunctionalAssessment, ADL, 
        MemoryComplaints, BehavioralProblems, Confusion, etc.
        
        ** Physiologiques (8)**: Age, BMI, SystolicBP, 
        DiastolicBP, Cholestérol complet
        
        ** Mode de vie (5)**: DietQuality, PhysicalActivity, 
        AlcoholConsumption, SleepQuality, Smoking
        
        ** Antécédents (7)**: FamilyHistory, CardiovascularDisease, 
        Diabetes, Depression, HeadInjury, etc.
        
        ** Démographiques (3)**: Gender, Ethnicity, EducationLevel
        """)
        
        st.success(f"""
        ** Fonctionnement Exact**  
        La pipeline fait exactement ceci:
        1. **Scaler**: 32 features → standardisation
        2. **Sélection**: 17 features optimales  
        3. **Random Forest**: Prédiction finale
        4. **Output**: Probabilité + Recommandation
        """)

st.markdown("---")
st.markdown("**Clinical Decision Support System")


# In[ ]:




