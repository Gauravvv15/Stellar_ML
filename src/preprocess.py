import pandas as pd
import numpy as np
import joblib


def clean_data(df):
    cols_drop=['object_ID','run_ID','rerun_ID','cam_col', 'field_ID','fiber_ID','spec_obj_ID', 'plate_ID']
    df.drop(cols_drop, axis=1, inplace=True)
    df=df.dropna()
    df.isna().sum()
    df['alpha']= pd.to_numeric(df['alpha'], errors='coerce')
    df['alpha']=df['alpha'].fillna(df['alpha'].median())
    return df

def modify_df(data):
    #creating new features
    data['u-g']=data['UV_filter'] - data['green_filter']     #this features are used in real astrophysics
    data['g-r']=data['green_filter'] - data['red_filter']
    data['r-i']=data['red_filter'] - data['near_IR_filter']
    data['i-z']= data['near_IR_filter'] - data['IR_filter'] 

    #flux ratios why: ratio normalize brightness / helps model generalize better 
    data['u_g_ratio']= data['UV_filter'] / (data['green_filter'] + 1e-5)
    data['g_r_ratio']= data['green_filter'] / (data['red_filter'] + 1e-5)
    data['r_i_ratio']= data['red_filter'] / (data['near_IR_filter'] + 1e-5)
    data['i_z_ratio']= data['near_IR_filter'] / (data['IR_filter'] + 1e-5)

    #log transform of brightness  why: Astronomical data is highly skewed / logs make pattern more linear, better for ml
    data['log_UV']=np.log1p(np.clip(data['UV_filter'], a_min=0, a_max=None))
    data['log_GREEN']=np.log1p(np.clip(data['green_filter'], a_min=0, a_max=None))
    data['log_RED']=np.log1p(data['red_filter'])
    data['log_IR']=np.log1p(np.clip(data['IR_filter'], a_min=0, a_max=None))
    data['log_near_IR']=np.log1p(data['near_IR_filter'])

    #total intensity
    data['total_flux']=(data['UV_filter'] + data['green_filter'] + data['red_filter'] + data['near_IR_filter'] + data['IR_filter'])  #represent all values recieved from object



    #normalized filters        #Removes scale differences / Focus on distribution of light
    data['UV_normalize']= data['UV_filter'] / data['total_flux']  
    data['IR_normalize']= data['IR_filter'] / data['total_flux']
    data['near_IR_normalize']=data['near_IR_filter'] / data['total_flux']

    return data

def align_columns(x):
    x_column=joblib.load('models/training_columns.pkl')

    aligned_x= x.reindex(columns=x_column, fill_value=0)
    return aligned_x
