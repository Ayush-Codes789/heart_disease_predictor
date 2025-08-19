import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the trained pipeline from the pickle file
try:
    with open('heart_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    print("Error: 'heart_pipeline.pkl' not found. Make sure the model file is in the correct directory.")
    pipeline = None
except Exception as e:
    print(f"Error loading pickle file: {e}")
    pipeline = None

@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the form.
    Processes user input, makes a prediction, and displays the result.
    """
    if not pipeline:
        return render_template('index.html', error="Model is not loaded. Please check server logs.")

    try:
        # Get all form values and convert them to the correct data types
        # The order of features must match the order in the training data
        # Features used: ['age', 'sex', 'cp', 'chol', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        form_values = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['chol']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]

        # Define the column names in the correct order
        columns = ['age', 'sex', 'cp', 'chol', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        # Create a pandas DataFrame from the user input
        input_data = pd.DataFrame([form_values], columns=columns)
        
        # Use the pipeline to make a prediction
        prediction = pipeline.predict(input_data)
        prediction_proba = pipeline.predict_proba(input_data)

        # Determine the output text based on the prediction
        if prediction[0] == 1:
            probability = prediction_proba[0][1] * 100
            result_text = f"The model predicts a high probability ({probability:.2f}%) of heart disease."
        else:
            probability = prediction_proba[0][0] * 100
            result_text = f"The model predicts a low probability ({probability:.2f}%) of heart disease."
            
        return render_template('index.html', prediction_text=result_text, prediction=prediction[0])

    except (ValueError, TypeError):
        # Handle cases where form fields might be empty or have incorrect data
        error_message = "Invalid input. Please ensure all fields are filled correctly with numerical values."
        return render_template('index.html', error=error_message)
    except Exception as e:
        # Handle other potential errors during prediction
        error_message = f"An error occurred during prediction: {e}"
        return render_template('index.html', error=error_message)

if __name__ == "__main__":
    # To run the app, execute `python app.py` in your terminal
    # and then open http://127.0.0.1:5000 in your web browser.
    app.run(debug=True)

