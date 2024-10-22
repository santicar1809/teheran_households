import pandas as pd
import os
import re
import numpy as np
from src.preprocessing.load_data import load_datasets

data=load_datasets()

data.info()

data.head()


def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s1 = s1.replace(' ','_')
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def columns_transformer(data):
    #Pasamos las columnas al modo snake_case
    columns=data.columns
    new_cols=[]
    for i in columns:
        i=to_snake_case(i)
        new_cols.append(i)
    data.columns=new_cols
    print(data.columns)
    return data

def preprocess_data(data):
    '''This function will clean the data by setting removing duplicates, 
    formatting the column types, names and removing incoherent data. The datasets
    will be merged in one joined by the CustomerID''' 
    
    data = columns_transformer(data)
    data['area'] = data['area'].str.replace(',', '', regex=False)
    #Pasamos la columna Area a numeric
    data['area'] = pd.to_numeric(data['area'])
    data.drop_duplicates(inplace=True)
    data=data.drop(['address'],axis=1)

    # Preprocesing merged dataset
    
    path = './files/datasets/intermediate/'

    if not os.path.exists(path):
        os.makedirs(path)

    data.to_csv(path+'preprocessed_data.csv', index=False)

    print(f'Dataframe created at route: {path}preprocessed_data.csv ')

    return data