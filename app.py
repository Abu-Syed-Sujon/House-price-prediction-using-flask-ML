from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np

app = Flask(__name__)

# File paths for your saved models
MODEL_PATHS = {
    'linear': 'models/linear_regression_model.pkl',
    'random_forest': 'models/random_forest_model.pkl',
    'gradient_boosting': 'models/gradient_boosting_model.pkl',
    'xgboost':'models/xgboost_model.pkl'
}

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_choice = request.form['model']
        
        # Numeric features
        num_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
        input_data = {}

        for col in num_cols:
            val = float(request.form[col])
            input_data[col] = val

        # Binary categorical features (yes/no)
        binary_cat_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        for col in binary_cat_cols:
            val = request.form[col].lower()
            input_data[col] = 1 if val == 'yes' else 0

        # One-hot encode furnishingstatus
        furnishing_status = request.form['furnishingstatus'].lower()
        furnishing_options = ['furnished', 'semi-furnished', 'unfurnished']
        for option in furnishing_options:
            col_name = f'furnishingstatus_{option}'
            input_data[col_name] = 1 if furnishing_status == option else 0

        # Align input with model feature order
        feature_order = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
                         'mainroad', 'guestroom', 'basement', 'hotwaterheating',
                         'airconditioning', 'prefarea',
                         'furnishingstatus_semi-furnished',
                         'furnishingstatus_unfurnished']

        final_input = np.array([[input_data[feat] for feat in feature_order]])

        # Load model with joblib
        model = joblib.load(MODEL_PATHS[model_choice])
        
        # Predict
        prediction = model.predict(final_input)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Predicted House Price: ${output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)
