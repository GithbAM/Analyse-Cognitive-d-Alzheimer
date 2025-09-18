# Pipeline d'Évaluation Cognitive Alzheimer ML

> Système d'aide au diagnostic médical avec **91,03% F1-Score** et dashboard Streamlit interactif

## Démo Live

**Dashboard :** [Clinical Decision Support]([https://share.streamlit.io](https://analyse-cognitive-d-alzheimer.streamlit.app/))

## Résultats

| Modèle | Accuracy | Précision | Rappel | **F1-Score** |
|--------|----------|-----------|--------|--------------|
| **Random Forest (Optimisé)** | **93,72%** | **91,95%** | **90,13%** | **🎯 91,03%** |
| SVM (Optimisé) | 84,65% | 80,28% | 75,00% | 77,55% |
| Régression Logistique (Optimisé) | 82,09% | 77,37% | 69,74% | 73,36% |

**Signification Clinique :**
- **Sensibilité élevée (90,13%)** : Excellente détection des cas positifs
- **Précision élevée (91,95%)** : Minimise les faux positifs
- **Performance équilibrée** : F1-Score de **91,03%** pour une aide au diagnostic fiable

## Stack Technique & Compétences

**ML Ops & Data Science:**
- `scikit-learn` (Random Forest, SVM, GridSearchCV)
- `pandas` / `numpy` (Traitement des données médicales)
- Feature Engineering Multi-méthodes (Corrélation, Random Forest, ANOVA F-test)
- Validation Croisée Stratifiée & Optimisation Hyperparamètres

**Déploiement & Production:**
- `streamlit` (Dashboard interactif)
- Déploiement Cloud (Streamlit Cloud)

## Fonctionnalités Clés

- **Traitement de Données Médicales** : Preprocessing spécialisé pour données d'évaluation clinique et cognitive
- **Sélection Multi-Méthodes de Features** : Combine analyse de corrélation, importance Random Forest et tests ANOVA F
- **Optimisation Avancée de Modèles** : Réglage automatique d'hyperparamètres avec validation croisée
- **Interprétabilité Clinique** : Focus sur l'IA explicable pour l'aide à la décision médicale
- **Pipeline Robuste** : Architecture modulaire adaptée au déploiement clinique
- **Dashboard Interactif** : Prédictions en temps réel pour les professionnels de santé

## Structure du Projet

```
alzheimer-ml-pipeline/
├── notebooks/           # Analyse EDA + Pipeline ML
├── src/                # Code modulaire + Dashboard
├── data/               # Dataset médical (2,149 patients)
├── models/             # Modèle entraîné (.pkl)
└── reports/            # Documentation technique
```

## Utilisation

```python
# Pipeline automatisé complet
pipeline = AlzheimerAnalysisPipeline("data.csv")
pipeline.run_complete_analysis()
best_model = pipeline.get_best_model()
predictions = pipeline.predict_new_data(patient_data)
```
