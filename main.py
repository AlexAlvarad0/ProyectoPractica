from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask (__name__)
with open ('modelo.pk1', 'rb') as f:
    model =       pickle.load(f)

@app.route('/predict', methods=['POST'])

def predict():
    try:
        data=request.json
        df=pd.Dataframe(data)

        predictions =model.predict(df).tolist()
        return jsonify ({'predictions': predictions})
    except Exception as e:
        return jsonify ({'error':str(e)})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)