from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pandas as pd

def feature_engineer(data):
    seed=12345
    df_train,df_test=train_test_split(data,test_size=0.7,random_state=seed) 
    scaler=MinMaxScaler()
    numeric=['area', 'price(usd)','room']
    df_train_scaled=scaler.fit_transform(df_train[numeric])
    df_train_scaled=pd.DataFrame(df_train_scaled,columns=numeric,index=df_train.index)
    df_train[numeric]=df_train_scaled
    return df_train,df_test