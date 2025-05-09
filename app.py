from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__, static_folder='static')

# Load the pre-trained models
lifestyle_model = joblib.load('lifestyle_model.pkl')
clinical_model = joblib.load('clinical_model.pkl')
combined_model = joblib.load('heart_attack_model.pkl')

# Homepage route
@app.route('/')
def home():
    return render_template('combinedui.html')

# Routes to display form pages
@app.route('/show_lifestyle')
def show_lifestyle():
    return render_template('lifestyle.html')

@app.route('/show_clinical')
def show_clinical():
    return render_template('variable.html')

@app.route('/show_combined')
def show_combined():
    return render_template('combined.html')

@app.route('/show_ecg')
def show_ecg():
    return render_template('combinedui/ecg.html')

@app.route('/account')
def show_account():
    return render_template('account.html')

# Utility function to safely get and convert inputs
def get_input_list(keys, form):
    try:
        return [float(form.get(key)) if form.get(key) is not None else 0 for key in keys]
    except ValueError as e:
        raise ValueError(f"Non-numeric input encountered: {e}")

# Predict route for lifestyle.html - Modified to return JSON for AJAX
@app.route('/predict_lifestyle', methods=['POST'])
def predict_lifestyle():
    try:
        # Map form field names to model feature names
        form_data = {
            'Age': float(request.form.get('age', 0)),
            'Gender': float(request.form.get('gender', 0)),
            'Occupation': float(request.form.get('occupation', 0)),
            'Sleep Duration': float(request.form.get('sleep_duration', 0)),
            'Quality of Sleep': float(request.form.get('quality_of_sleep', 0)),
            'Physical Activity Level': float(request.form.get('physical_activity', 0)),
            'Stress Level': float(request.form.get('stress_level', 0)),
            'BMI Category': float(request.form.get('bmi_category', 0)),
            'Heart Rate': float(request.form.get('heart_rate', 0)),
            'Daily Steps': float(request.form.get('daily_steps', 0)),
            'Blood Pressure (Systolic)': float(request.form.get('systolic', 0)),
            'Blood Pressure (Diastolic)': float(request.form.get('diastolic', 0))
        }
        
        # Create feature array in the correct order
        feature_keys = [
            'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
            'Physical Activity Level', 'Stress Level', 'BMI Category',
            'Heart Rate', 'Daily Steps', 'Blood Pressure (Systolic)',
            'Blood Pressure (Diastolic)'
        ]
        
        input_data = np.array([[form_data[key] for key in feature_keys]])
        
        # Make prediction
        prediction = lifestyle_model.predict(input_data)[0]
        
        # Convert binary prediction to sleep disorder categories (for demo purposes)
        # You can adjust this logic based on your specific model output
        sleep_disorder = 0  # Default: No Sleep Issues
        
        if prediction == 1:  # High Risk
            # Determine sleep disorder category based on stress and sleep quality
            stress_level = form_data['Stress Level']
            sleep_quality = form_data['Quality of Sleep']
            
            if stress_level > 7 or sleep_quality < 4:
                sleep_disorder = 2  # Severe Sleep Disorder
            else:
                sleep_disorder = 1  # Mild Sleep Disorder
        
        # Return JSON response for the AJAX request
        return jsonify({
            'prediction': int(prediction),
            'risk': "High Risk" if prediction == 1 else "Low Risk",
            'sleep_disorder': sleep_disorder
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Other routes remain the same...

@app.route('/predict_clinical', methods=['POST'])
def predict_clinical():
    try:
        feature_keys = [
            'Age', 'Gender', 'Chest Pain', 'RestingBP', 'FBS', 'RestingECG',
            'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'CA', 'Thal', 'Cholesterol'
        ]
        input_data = np.array([get_input_list(feature_keys, request.form)])
        prediction = clinical_model.predict(input_data)[0]
        result = "High Risk" if prediction == 1 else "Low Risk"
        return jsonify({'risk': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    #     return render_template('variable.html', prediction=result)
    # except Exception as e:
    #     return f"Error: {e}"

@app.route('/predict_combined', methods=['POST'])
def predict_combined():
    try:
        feature_keys = [
            'Age', 'Gender', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Family History',
            'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week',
            'Diet Type', 'Previous Heart Problems', 'Medication Use', 'Stress Level',
            'Sedentary Hours Per Day', 'Income', 'BMI', 'Physical Activity Days Per Week',
            'Sleep Hours Per Day'
        ]
        input_data = np.array([get_input_list(feature_keys, request.form)])
        prediction = combined_model.predict(input_data)[0]
        result = "High Risk" if prediction == 1 else "Low Risk"
        return jsonify({'risk': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)