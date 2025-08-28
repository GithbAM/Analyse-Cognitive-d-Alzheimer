# Pipeline d'Évaluation Cognitive Alzheimer ML

## Aperçu

Un pipeline complet de machine learning pour la prédiction de la maladie d'Alzheimer utilisant des données d'évaluation cognitive et clinique. Ce projet démontre des méthodologies avancées de data science pour les applications de santé, atteignant 91,03% de F1-Score avec optimisation Random Forest.

## Fonctionnalités Clés

- **Traitement de Données Médicales** : Preprocessing spécialisé pour données d'évaluation clinique et cognitive
- **Sélection Multi-Méthodes de Features** : Combine analyse de corrélation, importance Random Forest et tests ANOVA F
- **Optimisation Avancée de Modèles** : Réglage automatique d'hyperparamètres avec validation croisée
- **Interprétabilité Clinique** : Focus sur l'IA explicable pour l'aide à la décision médicale
- **Pipeline Robuste** : Architecture modulaire adaptée au déploiement clinique

## Structure du Projet

```
detection-alzheimer/
├── data/
│   └── alzheimers_disease_data.csv  # Dataset Clinique
├── notebooks/
│   ├── 01_Alzheimer_cognitive_cell.ipynb  # Analyse détaillée
│   ├── 02_Alzheimer_Cognitive_fonctions.ipynb  # Module de Fonctions Principales
│   └── 03_Alzheimer_Cognitive_pipeline.ipynb  # Pipeline ML Complet
├── reports/
│   ├── 01_Alzheimer_cognitive_cell.pdf  # Rapport détaillé
│   ├── 02_Alzheimer_Cognitive_fonctions.pdf  # Documentation des fonctions
│   └── 03_Alzheimer_Cognitive_pipeline.pdf  # Rapport du pipeline
├── src/
│   └── Alzheimer_Cognitive_fonctions.py  # Module de Fonctions Principales
├── .gitignore
└── README.md
```

## Implémentation Technique

### Traitement des Données
- **Validation de Données Cliniques** : Évaluation complète de qualité pour l'intégrité des données médicales
- **Détection d'Outliers** : Détection basée sur IQR identifiant 1 895 outliers potentiels
- **Mise à l'Échelle des Features** : Normalisation StandardScaler pour mesures cliniques
- **Gestion des Valeurs Manquantes** : Zéro valeur manquante maintient l'intégrité des données

### Feature Engineering
- **Sélection Multi-Modale** : 17 features sélectionnées parmi 32 variables originales
- **Pertinence Clinique** : Focus sur évaluations cognitives (MMSE, ADL, Évaluation Fonctionnelle)
- **Marqueurs Physiologiques** : Intégration d'indicateurs cardiovasculaires et métaboliques
- **Indicateurs Comportementaux** : Évaluation des plaintes mémorielles et problèmes comportementaux

### Pipeline Machine Learning
- **Comparaison de Modèles** : Régression Logistique, Random Forest, SVM
- **Optimisation d'Hyperparamètres** : Recherche sur grille avec validation croisée 5-fold
- **Métriques d'Évaluation** : Précision, precision, rappel, F1-score avec matrices de confusion
- **Validation Clinique** : Échantillonnage stratifié maintenant l'équilibre diagnostique

## Résultats

| Modèle | Accuracy | Précision | Rappel | **F1-Score** |
|--------|----------|-----------|--------|--------------|
| **Random Forest (Optimisé)** | **93,72%** | **91,95%** | **90,13%** | **🎯 91,03%** |
| SVM (Optimisé) | 84,65% | 80,28% | 75,00% | 77,55% |
| Régression Logistique (Optimisé) | 82,09% | 77,37% | 69,74% | 73,36% |

### Pourquoi le F1-Score est la métrique clé ?

En diagnostic médical, le **F1-Score** est essentiel car :
- ⚖️ **Équilibre critique** : Combine précision et rappel pour éviter les biais
- 🚨 **Coût des erreurs** : Faux négatifs (manquer Alzheimer) et faux positifs (sur-diagnostic) sont tous deux critiques
- 📊 **Dataset déséquilibré** : Gère mieux le ratio 64,6% sains / 35,4% Alzheimer que l'accuracy seule

**Signification Clinique :**
- **Sensibilité élevée (90,13%)** : Excellente détection des cas positifs
- **Précision élevée (91,95%)** : Minimise les faux positifs 
- **Performance équilibrée** : F1-Score de 91,03% indique un modèle fiable pour l'aide au diagnostic

## Features Clés Identifiées

**Variables Prédictives Principales :**
1. **FunctionalAssessment** - Capacités de vie quotidienne
2. **ADL (Activities of Daily Living)** - Mesures d'indépendance
3. **MMSE** - Scores Mini-Mental State Examination
4. **MemoryComplaints** - Préoccupations cognitives subjectives
5. **BehavioralProblems** - Symptômes psychologiques

## Compétences Techniques Démontrées

- **Data Science Santé** : Traitement de données médicales, feature engineering clinique
- **ML Avancé** : Comparaison multi-modèles, optimisation d'hyperparamètres
- **Analyse Statistique** : Sélection de features, analyse de corrélation, tests ANOVA
- **Visualisation de Données** : Exploration de données cliniques, visualisation de performance de modèles
- **Écosystème Python** : scikit-learn, pandas, numpy, matplotlib, seaborn

## Utilisation

```python
# Initialiser et exécuter le pipeline complet
pipeline = AlzheimerAnalysisPipeline("alzheimers_disease_data.csv")

# Exécuter l'analyse complète
pipeline.run_complete_analysis(
    remove_outliers_flag=False,
    optimize_hyperparams=True,
    show_plots=True
)

# Obtenir le meilleur modèle pour prédictions
best_model = pipeline.get_best_model()
predictions, probabilities = pipeline.predict_new_data(new_patient_data)
```

## Étapes du Pipeline

1. **Chargement & Preprocessing** - Validation et nettoyage des données cliniques
2. **Évaluation de Qualité** - Évaluation complète de la qualité des données
3. **Analyse Exploratoire** - Analyse de distribution et études de corrélation
4. **Division des Données** - Split train/test stratifié avec mise à l'échelle
5. **Sélection de Features** - Analyse multi-méthodes d'importance des features
6. **Entraînement de Modèles** - Évaluation des modèles de base
7. **Optimisation d'Hyperparamètres** - Optimisation par recherche sur grille
8. **Analyse des Résultats** - Comparaison de performance et interprétation

## Applications Cliniques

Le pipeline soutient les professionnels de santé dans :
- Détection précoce et dépistage d'Alzheimer
- Évaluation des risques
- Systèmes d'aide à la décision clinique


## Caractéristiques du Dataset

- **Taille d'Échantillon** : 2 149 patients
- **Features** : 35 variables cliniques et cognitives
- **Distribution des Classes** : 64,6% sains, 35,4% diagnostic Alzheimer
- **Qualité des Données** : Dataset complet sans valeurs manquantes

## Prérequis

- Python 3.8+
- scikit-learn, pandas, numpy
- matplotlib, seaborn
- Module de fonctions personnalisées : `Alzheimer_Cognitive_fonctions.py`
