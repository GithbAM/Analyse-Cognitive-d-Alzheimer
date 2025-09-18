#!/usr/bin/env python
# coding: utf-8

# In[1]:


# train_and_save.py
from Alzheimer_Cognitive_pipeline import AlzheimerAnalysisPipeline
import joblib

pipeline = AlzheimerAnalysisPipeline("alzheimers_disease_data.csv")
pipeline.run_complete_analysis(remove_outliers_flag=False, optimize_hyperparams=True, show_plots=False)

# Sauvegarde le pipeline entraîné
joblib.dump(pipeline, "alzheimer_pipeline_trained.pkl")
print("Pipeline entraîné et sauvegardé dans alzheimer_pipeline_trained.pkl")


# In[ ]:




