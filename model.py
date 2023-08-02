import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df =pd.read_csv('company_bankruptcy_prediction.csv')

def preprocess_inputs(df):
    data = df.copy()
    
    
    # Split df into X and y
    y = data['Bankrupt?']
    X = data.drop('Bankrupt?', axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test

def cbp_model(X_train,y_train):
    svc = SVC()
    svc.fit(X_train,y_train)
    return cbp_model
