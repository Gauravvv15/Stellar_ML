import pandas as pd
from src.preprocess import modify_df,align_columns
from src.predict import class_predict
import joblib

model=joblib.load('models/xgb_redshift_model.pkl')
alpha=float(input("Enter the 'alpha' value of the object: "))
delta=float(input("Enter the 'delta' value of the object: "))
UV_filter=float(input("Enter the 'Ultraviolet' value of the object: "))
green_filter=float(input("Enter the 'green_filter' value of the object: "))
red_filter=float(input("Enter the 'red_filter' value of the object: "))
near_IR_filter=float(input("Enter the 'near_IR_filter' value of the object: "))
IR_filter=float(input("Enter the 'IR_filter' value of the object: "))
MJD=float(input("Enter the 'MJD' value of the object: "))
print('Input data received, processing...\n')

cols=['alpha', 'delta', 'UV_filter', 'green_filter', 'red_filter', 'near_IR_filter', 'IR_filter', 'MJD']

input_data=[alpha, delta, UV_filter, green_filter, red_filter, near_IR_filter, IR_filter, MJD]

input_df=pd.DataFrame([input_data], columns=cols)
modified_input_df=modify_df(input_df)
modified_input_df=align_columns(modified_input_df)
modified_input_df.fillna(0, inplace=True) #handle any missing values that might arise from feature engineering

prediction=class_predict(modified_input_df)
print(prediction)
