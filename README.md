# 🏠 House Price Prediction Using Machine Learning

This project predicts the selling price of a house based on features like area, number of bedrooms, bathrooms, location advantages, and amenities using machine learning regression models.

## 🚀 Overview

This ML project applies supervised learning techniques to build a regression model that can estimate house prices. It includes:

- Data preprocessing (including handling categorical variables)
- Model training using algorithms like Random Forest, Linear Regression, etc.
- Feature scaling
- Model evaluation
- Deployment-ready structure (with optional web interface)

---

## 🧠 Features Used

The following features were used to train the model:

- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `parking`
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `prefarea`
- `furnishingstatus` (converted to multiple binary columns)

---

## 📂 Project Structure

house-price-prediction/
│
├── data/
│ └── housing.csv # Raw dataset
│
├── model/
│ ├── house_price_model.pkl # Trained ML model
│ └── scaler.joblib # Saved StandardScaler
│
├── app/
│ ├── app.py # (Optional) Flask or Django app
│ └── templates/
│ └── index.html # Frontend form to submit inputs

Python 3.8+

pandas

numpy

scikit-learn

joblib

Flask / Django (if deploying web interface)

matplotlib / seaborn (for data visualization)
