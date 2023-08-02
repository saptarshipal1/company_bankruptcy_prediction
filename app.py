import numpy as np
import pandas as pd
import pickle
from flask import Flask,request,render_template

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from model import preprocess_inputs, cbp_model



df = pd.read_csv('company_bankruptcy_prediction.csv')

preprocess_inputs(df)

print(X_train.shape,y_train.shape)

