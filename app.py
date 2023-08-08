import numpy as np
import pandas as pd
import pickle
from flask import Flask,request,render_template

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from model import preprocess_inputs, cbp_model

app = Flask(__name__,template_folder='template')

df = pd.read_csv('company_bankruptcy_prediction.csv')
model = pickle.load(open("cbp_best_model.pkl","rb"))

@app.route('/',methods=['GET','POST'])  

def predict_datapoint():
                if request.method=='GET':
                    return render_template('index_v_2.html')
                else:
                    X_train, X_test, y_train, y_test = preprocess_inputs(df)
                    
                    pca = PCA(n_components= 10)
                    pca.fit(X_train)

                    X_train_reduced = pd.DataFrame(pca.transform(X_train), index=X_train.index, columns=["PC" + str(i) for i in range(1,11)])

                    PC1 = float(request.form["PC1"])
                    PC2 = float(request.form["PC2"])
                    PC3 = float(request.form["PC3"])
                    PC4 = float(request.form["PC4"])
                    PC5 = float(request.form["PC5"])
                    PC6 = float(request.form["PC6"])
                    PC7 = float(request.form["PC7"])
                    PC8 = float(request.form["PC8"])
                    PC9 = float(request.form["PC9"])
                    PC10 = float(request.form["PC10"])
                    
                    sample_data = pd.DataFrame([[PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10]],columns=["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10"])
                    
                    model.fit(X_train_reduced,y_train)

                    results = model.predict(sample_data)
                    
                    if results[0] == 0:
                        return render_template('index_v_2.html',results= 'company is not bankrupt')
                    
                    else: return render_template('index_v_2.html',results= 'company is bankrupt')
                    
                    

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=9696)
    
