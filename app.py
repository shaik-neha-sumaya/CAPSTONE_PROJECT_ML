
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('best_car_model.pkl', 'rb') as f:
    model = pickle.load(f)

labels = ['unacc', 'acc', 'good', 'vgood']

# Manual encoding (without external label encoder)
buying_map = {'vhigh':0, 'high':1, 'med':2, 'low':3}
maint_map = {'vhigh':0, 'high':1, 'med':2, 'low':3}
doors_map = {'2':0, '3':1, '4':2, '5more':3}
persons_map = {'2':0, '4':1, 'more':2}
lug_boot_map = {'small':0, 'med':1, 'big':2}
safety_map = {'low':0, 'med':1, 'high':2}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [
                buying_map[request.form['buying']],
                maint_map[request.form['maint']],
                doors_map[request.form['doors']],
                persons_map[request.form['persons']],
                lug_boot_map[request.form['lug_boot']],
                safety_map[request.form['safety']]
            ]
            features = np.array(features).reshape(1, -1)
            prediction = model.predict(features)[0]
            result = labels[prediction]
            return render_template('result.html', prediction_text=result)
        except Exception as e:
            return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
