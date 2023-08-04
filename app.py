import numpy as np
import pandas as pd
import pickle
from flask import Flask,request,render_template

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from model import preprocess_inputs, cbp_model



df = pd.read_csv('company_bankruptcy_prediction.csv')

X_train, X_test, y_train, y_test = preprocess_inputs(df)

model = pickle.load(open("cbp_best_model.pkl","rb"))

sample_data = X_test.sample()

print('The sample data is \n' + sample_data)

model = pickle.load(open("cbp_best_model.pkl","rb"))

model.fit(X_train,y_train)

model.predict(sample_data)

