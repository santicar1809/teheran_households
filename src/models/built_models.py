import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from src.models.hyper_parameters import all_models
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def model_structure(data, pipeline, param_grid):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''
    seed=12345
    features=data.drop(['price(usd)'],axis=1)
    target=data['price(usd)']
    features_train,features_valid,target_train,target_valid=train_test_split(features,target,test_size=0.25,random_state=seed)   
    # Training the model
    gs = GridSearchCV(pipeline, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    gs.fit(features_train,target_train)

    # Scores
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    rmse_val,r2_val = eval_model(best_estimator,features_valid,target_valid)
    print(f'RMSE: {rmse_val}')
    
    results = best_estimator, best_score,rmse_val,r2_val
    return results
    
def eval_model(best,features_valid,target_valid):
    random_prediction = best.predict(features_valid)
    random_rmse=mean_squared_error(target_valid,random_prediction)**0.5
    r2=r2_score(target_valid,random_prediction)
    print("RMSE:",random_rmse)
    print("R2:",r2)
    return random_rmse,r2

## Network Model Structure

def build_model(data):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(data,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),  # Dropout for regularization
        Dense(64, activation='relu', input_shape=(data,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #Dropout(0.3),  # Dropout for regularization
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #Dropout(0.3),  # More dropout for regularization        
        Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #Dropout(0.3),  # More dropout for regularization        
        Dense(1, activation='linear')
    ])
    return model

def tens_flow(data):
    
    seed=12345    
    features=data.drop(['price(usd)'],axis=1)
    target=data['price(usd)']
    features_train,features_valid,target_train,target_valid=train_test_split(features,target,random_state=seed,test_size=0.2)
    
    # Compiling the model
    model = build_model(features_train.shape[1])
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    
    model.summary()
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Training the model using GPU if available
    with tf.device('/GPU:0'):  
        history = model.fit(features_train, target_train, epochs=200, batch_size=32, 
                            validation_data=(features_valid, target_valid), callbacks=[early_stopping])

    # Evaluating the model
    y_pred = model.predict(features_valid)
    rmse_score = mean_squared_error(target_valid, y_pred)
    r2=r2_score(target_valid,y_pred)
    print(f"RMSE Score: {rmse_score}")
    results = ['Keras',rmse_score,r2]
    results_df = pd.DataFrame({'model':[results[0]],'rmse_score':[results[1]],'r2_score':[results[2]]})

    return results_df,model


def iterative_modeling(data):
    '''This function will bring the hyper parameters from all_model() 
    and wil create a complete report of the best model, estimator, 
    score and validation score'''

    models = all_models() 

    output_path = './files/modeling_output/model_fit/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    tf_results = tens_flow(data) 
    #joblib.dump(tf_results[1],output_path +f'best_random_nn.joblib')
    tf_results[1].save(output_path+'best_random_nn.h5')
    # Concatening logistic models and neuronal network


    results = []
    # Iterating the models
    for model in models:
        best_estimator, best_score, rmse_val,r2_val= model_structure(data, model[1], model[2])
        results.append([model[0],best_estimator, rmse_val,r2_val])      
        # Guardamos el modelo
        joblib.dump(best_estimator,output_path +f'best_random_{model[0]}.joblib')
    results_df = pd.DataFrame(results, columns=['model','best_estimator','rmse_score','r2_score'])

    final_rev = pd.concat([results_df,tf_results[0]])

    final_rev.to_csv('./files/modeling_output/reports/model_report.csv',index=False)
    final_rev_sum=final_rev[['model','rmse_score','r2_score']]
    return final_rev_sum


