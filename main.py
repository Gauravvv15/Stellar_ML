import numpy as np
import pandas as pd
from src.preprocess import modify_df,align_columns
from src.predict import class_predict
import joblib

def safe_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter numeric values only.")

while True:
    user=input('Do you want to classify an astronomical object? (yes/no): ')
    if user.lower() in ['no', 'n', 'stop', 'exit', 'quit']:
        break
    else:
        try:
            model=joblib.load('models/xgb_redshift_model.pkl')
            
            alpha=safe_float("Enter the 'alpha' value of the object: ")
            delta=safe_float("Enter the 'delta' value of the object: ")
            UV_filter=safe_float("Enter the 'Ultraviolet' value of the object: ")
            green_filter=safe_float("Enter the 'green_filter' value of the object: ")
            red_filter=safe_float("Enter the 'red_filter' value of the object: ")
            near_IR_filter=safe_float("Enter the 'near_IR_filter' value of the object: ")
            IR_filter=safe_float("Enter the 'IR_filter' value of the object: ")

            print('Input data received, processing...\n')

            cols=['alpha', 'delta', 'UV_filter', 'green_filter', 'red_filter', 'near_IR_filter', 'IR_filter']

            input_data=[alpha, delta, UV_filter, green_filter, red_filter, near_IR_filter, IR_filter]

            input_df=pd.DataFrame([input_data], columns=cols)
            modified_input_df=modify_df(input_df)
            modified_input_df=align_columns(modified_input_df)
            modified_input_df.fillna(0, inplace=True) #handle any missing values that might arise from feature engineering

            prediction=class_predict(modified_input_df)
            print(prediction)

        except Exception as e:
            print(f"error: {e}")
            print("\nAn Error occured while processing your input. Please insure all inputs are valid numbers and try again.\n")
            continue