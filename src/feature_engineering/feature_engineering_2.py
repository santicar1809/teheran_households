from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from src.preprocessing.load_data import load_datasets
from src.preprocessing.preprocess import preprocess_data
data = load_datasets() #Loading stage
data = preprocess_data(data)
data=data
def binary_transform(column):
    if column==True:
        return 1
    elif column==False:
        return 0

#def feature_engineer(data):
seed=12345
data[['elevator','warehouse','parking']]=data[['elevator','warehouse','parking']].astype('int')
df_train,df_test=train_test_split(data,test_size=0.8,random_state=seed)

unseen_labels = set(df_test['address']) - set(df_train['address'])
if unseen_labels:
    print(f"Unseen labels in test set: {unseen_labels}")
address_count=data['address'].value_counts()
address_count

encoder=LabelEncoder()
df_train['address'] = encoder.fit_transform(df_train['address'])
df_test['address'] = encoder.transform(df_test['address'])

#return data