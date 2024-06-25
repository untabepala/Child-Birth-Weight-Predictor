from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

#loading models
rid = pickle.load(open('rid.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))
#creating flask app
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        Age_years = request.form['Age_years']
        Height_cm = request.form['Height_cm']
        Parity = request.form['Parity']
        ANC = request.form['ANC']
        Iwt_kg = request.form['Iwt_kg']
        FWt_kg = request.form['FWt_kg']
        IBP_sys = request.form['IBP_sys']
        IBP_dias = request.form['IBP_dias']
        FBP_sys = request.form['FBP_sys']
        FBP_dias = request.form['FBP_dias']
        IHb_gm = request.form['IHb_gm']
        FHb_gm = request.form['FHb_gm']
        BS_RBS = request.form['BS_RBS']
        LNH = request.form['LNH']
        Bgroup = request.form['Bgroup']
        Term_Preterm = request.form['Term_Preterm']

        features = np.array([[Age_years, Height_cm, Parity, ANC, Iwt_kg, FWt_kg, IBP_sys, IBP_dias, FBP_sys, FBP_dias, IHb_gm, FHb_gm, BS_RBS, LNH, Bgroup, Term_Preterm]],
                            dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = rid.predict(transformed_features).reshape(1, -1)

        return render_template('index.html', prediction=prediction)



# python main
if __name__=='__main__':
    app.run(debug=True)