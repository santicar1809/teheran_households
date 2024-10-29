from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def feature_engineer_2(data):
    seed=12345
    data[['elevator','warehouse','parking']]=data[['elevator','warehouse','parking']].astype('int')
    data=data.drop(['cluster'],axis=1)
    df_train,df_test=train_test_split(data,test_size=0.3,random_state=seed)
    encoder=LabelEncoder()
    df_train['address'] = encoder.fit_transform(df_train['address'])
     # Las etiquetas no vistas se convierten en -1 (o puedes manejarlo como prefieras)

    df_test['address'] = df_test['address'].apply(lambda x: encoder.transform([x])[0] 
                                                          if x in encoder.classes_ else -1)
    output_path = './test/files/'
    df_test.to_csv(output_path+'df_test.csv',index=False)

    return df_train