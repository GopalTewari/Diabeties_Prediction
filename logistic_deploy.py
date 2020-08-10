#Let's start with importing necessary libraries
import pickle
import numpy as np
import pandas as pd

class predObj:

    def predict_log(self, Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
        print("here")
        with open("Model\sandardScalar.sav", 'rb') as f:
            scalar = pickle.load(f)

        with open("Model\modelForPrediction.sav", 'rb') as f:
            model = pickle.load(f)
        scaled_data = scalar.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        print(scaled_data)
        predict = model.predict([scaled_data])
        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'

        return result



