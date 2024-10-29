import sys
from src.preprocessing.load_data import load_datasets
from src.preprocessing.preprocess import preprocess_data
from src.EDA.EDA import eda_report
from src.feature_engineering.features_engineer import feature_engineer
from src.feature_engineering.feature_engineering_2 import feature_engineer_2
from src.models.clustering_models import clustering_data
from src.models.built_models import iterative_modeling
import pandas as pd

def main():
    '''This main function progresses through various stages to process data, 
    evaluate variables, and create a robust model for predicting churned users. 
    For more detailed information, please refer to the README.md file. '''
    data = load_datasets() #Loading stage
    preprocessed_data = preprocess_data(data) #Preprocessing stage
    eda_report(preprocessed_data) # Analysis stage
    processed_data = feature_engineer(preprocessed_data) # Feature engineering stage
    clustering_data(processed_data)
    model_data=feature_engineer_2(preprocessed_data)
    results = iterative_modeling(model_data) # Modeling stage
    results.to_csv('./files/modeling_output/reports/summarized_result.csv',index=False)
    return results

results = main()
