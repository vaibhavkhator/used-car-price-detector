import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data_path = r"C:\Users\khato\Downloads\cardekho_dataset.csv"
df = pd.read_csv(data_path)

for col in ['car_name', 'Unnamed: 0']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

X = df.drop(columns=['selling_price'])
y = df['selling_price']

categorical_cols = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
numerical_cols = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

joblib.dump(model, "used_car_price_model.pkl")

def predict_price():
    try:
        user_data = {
            'brand': brand_entry.get().strip(),
            'model': model_entry.get().strip(),
            'seller_type': seller_type_combobox.get(),
            'fuel_type': fuel_type_combobox.get(),
            'transmission_type': transmission_combobox.get(),
            'vehicle_age': int(vehicle_age_entry.get()),
            'km_driven': int(km_driven_entry.get()),
            'mileage': float(mileage_entry.get()),
            'engine': int(engine_entry.get()),
            'max_power': float(max_power_entry.get()),
            'seats': int(seats_entry.get())
        }

        input_df = pd.DataFrame([user_data])
        
        predicted_price = model.predict(input_df)[0]
        
        result_label.config(text=f"Estimated Selling Price: ₹{int(predicted_price):,}", foreground="#006400")
        
    except ValueError as ve:
        messagebox.showerror("Input Error", "Please ensure all numerical fields contain valid numbers.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

root = tk.Tk()
root.title("Used Car Price Predictor")
root.geometry("500x600")
root.configure(bg="#f0f0f0")

title_label = tk.Label(root, text="Used Car Price Predictor", font=("Helvetica", 20, "bold"), bg="#f0f0f0", fg="#333333")
title_label.pack(pady=20)

frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
frame.pack(padx=20, pady=10, fill="x")

fields = [
    ("Brand", tk.Entry, {}),
    ("Model", tk.Entry, {}),
    ("Seller Type", ttk.Combobox, {"values": ["Dealer", "Individual"]}),
    ("Fuel Type", ttk.Combobox, {"values": ["Petrol", "Diesel", "CNG", "LPG", "Electric"]}),
    ("Transmission Type", ttk.Combobox, {"values": ["Manual", "Automatic"]}),
    ("Vehicle Age (years)", tk.Entry, {}),
    ("Kilometers Driven", tk.Entry, {}),
    ("Mileage (kmpl)", tk.Entry, {}),
    ("Engine (CC)", tk.Entry, {}),
    ("Max Power (bhp)", tk.Entry, {}),
    ("Number of Seats", tk.Entry, {})
]

entries = {}
for i, (label_text, widget_type, kwargs) in enumerate(fields):
    label = tk.Label(frame, text=label_text, font=("Helvetica", 12), bg="#ffffff", fg="#555555")
    label.grid(row=i, column=0, padx=10, pady=5, sticky="w")
    
    widget = widget_type(frame, font=("Helvetica", 12), **kwargs)
    widget.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
    entries[label_text] = widget

brand_entry = entries["Brand"]
model_entry = entries["Model"]
seller_type_combobox = entries["Seller Type"]
fuel_type_combobox = entries["Fuel Type"]
transmission_combobox = entries["Transmission Type"]
vehicle_age_entry = entries["Vehicle Age (years)"]
km_driven_entry = entries["Kilometers Driven"]
mileage_entry = entries["Mileage (kmpl)"]
engine_entry = entries["Engine (CC)"]
max_power_entry = entries["Max Power (bhp)"]
seats_entry = entries["Number of Seats"]

seller_type_combobox.set("Dealer")
fuel_type_combobox.set("Petrol")
transmission_combobox.set("Manual")

predict_button = tk.Button(root, text="Predict Price", command=predict_price, font=("Helvetica", 14, "bold"), 
                          bg="#4CAF50", fg="white", activebackground="#45a049", relief="flat")
predict_button.pack(pady=20)

result_label = tk.Label(root, text="", font=("Helvetica", 16, "bold"), bg="#f0f0f0", fg="#006400")
result_label.pack(pady=10)

frame.columnconfigure(1, weight=1)

root.mainloop()
