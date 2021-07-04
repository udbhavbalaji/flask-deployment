from logging import debug
from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    age = int(request.form['age'])
    head_size = int(request.form['head_size'])

    temp_gender = gender
    temp_age = age

    if gender.strip().lower() == 'male':
        gender = 1
    else:
        gender = 2

    if age <= 18:
        age = 2
    else:
        age = 1

    test_in = np.array([gender,age,head_size]).reshape(1,-1)
    pred_weight = model.predict(test_in)
    output = round(pred_weight[0], 2)
    return render_template('after.html', gender="Gender: {}".format(temp_gender), age="Age: {}".format(temp_age),
     head_size="Head Size: {}".format(head_size), prediction_text="Brain Weight is {} grams".format(output))

if __name__ == "__main__":
    app.run(debug=True)