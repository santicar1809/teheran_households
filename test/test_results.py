import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.keras.models import load_model
import joblib

def test_main():
    data=pd.read_csv('./test/files/df_test.csv')
    features=data.drop(['price(usd)'],axis=1)
    target=data['price(usd)']
    models_name=['dummy','Linreg','XGboost','lightgbm','Random_Forest','cat','dt']
    results=[]
    for model in models_name:    
        new_model = joblib.load(f'./files/modeling_output/model_fit/best_random_{model}.joblib')
        prob = new_model.predict(features)
        mse_score = mean_squared_error(target,prob)
        r2_val=r2_score(target,prob)
        results.append([model,mse_score,r2_val])

    results_df = pd.DataFrame(results, columns=['model','test_mse','test_r2'])

    #model_nn = joblib.load(f'./files/modeling_output/model_fit/best_random_nn.joblib')
    model_nn = load_model('./files/modeling_output/model_fit/best_random_nn.h5')
    proba=model_nn.predict(features)
    msen_score=mean_squared_error(target,proba)
    r2n_score=r2_score(target,proba)
    results_n=['Keras',msen_score,r2n_score]
    results_n_df = pd.DataFrame({'model':[results_n[0]],'test_mse':[results_n[1]],'test_r2':[results_n[2]]})
    final_rev = pd.concat([results_df,results_n_df])
    final_rev.to_csv('./files/modeling_output/reports/model_test_report.csv',index=False)
    print(results_n_df)
    return final_rev
    #return results_df
results_test = test_main()

pprint(results_test)