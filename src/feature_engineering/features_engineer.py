from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pandas as pd

def feature_engineer(data):
    seed=12345
    
    scaler=MinMaxScaler()
    numeric=['area', 'price(usd)','room']
    data_scaled=scaler.fit_transform(data[numeric])
    data_scaled=pd.DataFrame(data_scaled,columns=numeric,index=data.index)
    data[numeric]=data_scaled
    return data