import numpy as np
import pandas as pd
import pickle
from flask import Flask,request,render_template

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from model import preprocess_inputs, cbp_model

app = Flask(__name__)

df = pd.read_csv('company_bankruptcy_prediction.csv')
model = pickle.load(open("cbp_best_model.pkl","rb"))

@app.route('/',methods=['GET','POST'])  

def predict_datapoint():
                if request.method=='GET':
                    return render_template('index.html')
                else:
                    X_train, X_test, y_train, y_test = preprocess_inputs(df)

                    sample_data = X_test.sample()
                    
                    model.fit(X_train,y_train)

                    return render_template('index.html',random_input_data=sample_data[0])

                    results = model.predict(sample_data)
                    
                    return render_template('index.html',results=results[0])

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=9696)
    
