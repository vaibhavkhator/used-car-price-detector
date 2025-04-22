# 🚗 Used Car Price Predictor

A machine learning project with a Tkinter GUI to predict the **selling price of used cars** based on user input features like brand, model, mileage, etc.

## 📌 Features

- Predicts resale price of a used car
- User-friendly GUI using Tkinter
- Trained using `RandomForestRegressor`
- Includes both categorical and numerical features
- Preprocessing with `OneHotEncoder` and `ColumnTransformer`
- Model serialization with `joblib`

## 🗃 Dataset

- Dataset: `cardekho_dataset.csv`
- Make sure to update the path to your dataset in the script:
  ```python
  data_path = r"C:\path\to\your\cardekho_dataset.csv"
