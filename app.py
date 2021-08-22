import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

def convert_to_df(array):
    num_feats_test = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]
    df=pd.DataFrame([array],columns=num_feats_test)
    return df

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = convert_to_df(int_features)
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', interest_level='Interest level is:  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)