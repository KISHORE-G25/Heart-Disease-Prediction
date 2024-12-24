import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
  input_features = [float(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']

  df = pd.DataFrame(features_value, columns=features_name)
  output = model.predict(df)
  if output==1:
      res_val = "The Patient Have Heart Disease,please consult the Doctor"
  else:
      res_val = "The Patient is Normal"

    


  return render_template('index1.html', prediction_text='Result - {}'.format(res_val))

if __name__ == "__main__":
  app.run()
##host='0.0.0.0',debug=False, port = 4566
