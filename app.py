from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# diabetes
model_diabetes = pickle.load(open(r'healthcare_project\models\diabetes_model.pkl', 'rb'))
scalar_diabetes = pickle.load(open(r'healthcare_project\models\diabetes_scaler.pkl', 'rb'))

# heart
model_heart = pickle.load(open(r'healthcare_project\models\heart_model.pkl', 'rb'))
scalar_heart = pickle.load(open(r'healthcare_project\models\heart_scaler.pkl', 'rb'))

# parkinsons
model_parkinsons = pickle.load(open(r'healthcare_project\models\parkinsons_model.pkl', 'rb'))
scalar_parkinsons = pickle.load(open(r'healthcare_project\models\parkinsons_scalar.pkl', 'rb'))

# Routes 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diabetes', methods = ['GET', 'POST'])
def diabetes():
    result = None
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_input = scalar_diabetes.transform([features])
        prediction = model_diabetes.predict(final_input)[0]
        result = 'Diabetes Detected' if prediction == 1 else "Non-Diabetes"
    
    cm_plot = r'healthcare_project\plots\diabetes_cm.png'
    fi_plot = r'healthcare_project\plots\diabetes_features.png'
    
    return render_template('diabetes.html', result = result, cm_plot = cm_plot, fi_plot = fi_plot)

@app.route('/heart', methods = ['GET', 'POST'])
def heart():
    result = None
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_inputer = scalar_heart.transform([features])
        prediction = model_heart.predict(final_inputer)[0]
        result = 'Heart Disease Detected' if prediction == 1 else "No heart disease"
        
    cm_plot = r'healthcare_project\plots\heartdisease_cm.png'
    fi_plot = r'healthcare_project\plots\heartdisease_features.png'

    return render_template('heart.html', result = result, cm_plot = cm_plot, fi_plot = fi_plot)

@app.route('/parkinson', methods = ['GET', 'POST'])
def parkinson():
    result = None
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_input = scalar_parkinsons.transform([features])
        prediction = model_parkinsons.predict(final_input)[0]
        result = "Parkinson's Detected " if prediction == 1 else "No Parkinson's detected"
    
    cm_plot = r'healthcare_project\plots\parkinsons_cm.png'
    fi_plot = r'healthcare_project\plots\parkinsons_feature.png'
    
    return render_template('parkinson.html', result = result, cm_plot = cm_plot, fi_plot = fi_plot)

if __name__ == "__main__":
    app.run(debug=True)
    
