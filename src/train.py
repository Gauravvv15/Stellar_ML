import joblib
import os
import pandas as pd
import numpy as np
from preprocess import modify_df, clean_data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from xgboost import XGBRegressor, XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from preprocess import clean_data

BASE_DIR=os.path.dirname(os.path.dirname(__file__))
FILE_PATH=os.path.join(BASE_DIR, 'data','StarClassificationDataset.csv')


def split_data(data):
    x=data.drop(['class', 'red_shift'], axis=1)
    y=data['red_shift']

    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)

    joblib.dump(x_train.columns.tolist(), 'training_columns.pkl')

    return x_train, x_test, y_train, y_test

def classifier_model():
    # try:
    #     class_model=joblib.load('RandomForestClassifier_model.pkl')
    # except:
    #     print('Model is not train yet!')
    
    df=pd.read_csv('data/StarClassificationDataset.csv', low_memory=False)
    clean_data(df)
    df['alpha']= pd.to_numeric(df['alpha'], errors='coerce')
    df['alpha']=df['alpha'].fillna(df['alpha'].median())
    df=modify_df(df)

    df['max_flux']= df[['UV_filter', 'green_filter', 'red_filter', 'near_IR_filter', 'IR_filter']].max(axis=1)
    df['min_flux']= df[['UV_filter', 'green_filter', 'red_filter', 'near_IR_filter', 'IR_filter']].min(axis=1)
    df['flux_range']= df['max_flux'] - df['min_flux']

    x_class=df.drop(['class', 'red_shift'], axis=1)
    y_class=df['class']

    #use stratifiedshufflesplit
    split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for train_index, test_index in split.split(x_class, y_class):
        x_train_class=x_class.iloc[train_index]
        x_test_class=x_class.iloc[test_index]
        y_train_class=y_class.iloc[train_index]
        y_test_class=y_class.iloc[test_index]


    # x_train_class, x_test_class, y_train_class, y_test_class=split_data(df)

    # class_model=RandomForestClassifier(n_estimators=150,  #1
    #                              max_depth=None, 
    #                                 n_jobs=-1,
    #                              random_state=42,
    #                              class_weight='balanced')
    
    
    # 2. Random Forest with tuned hyperparameters
    class_model=RandomForestClassifier(
    n_estimators=800,
    max_depth=28,
    min_samples_split=15,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight={'GALAXY': 1, 'QSO': 1, 'STAR': 2},
    random_state=42,
    n_jobs=-1
    )

    class_model.fit(x_train_class, y_train_class)

    class_pred=class_model.predict(x_test_class)
    acc=accuracy_score(y_test_class, class_pred)
    print('randomforewst classifier results:')
    print(f'Classification Accuracy: {acc}')
    print(classification_report(y_test_class, class_pred))
    print(f'Confusion Matrix:\n{confusion_matrix(y_test_class, class_pred)}')

    joblib.dump(class_model, 'models/RandomForestClassifier_model.pkl')

    joblib.dump(x_train_class.columns.tolist(), 'models/classifier_columns.pkl')     


def xgb_classifier_model():
    df=pd.read_csv('data/StarClassificationDataset.csv', low_memory=False)
    clean_data(df)
    df['alpha']= pd.to_numeric(df['alpha'], errors='coerce')
    df['alpha']=df['alpha'].fillna(df['alpha'].median())
    df=modify_df(df)

    df['max_flux']= df[['UV_filter', 'green_filter', 'red_filter', 'near_IR_filter', 'IR_filter']].max(axis=1)
    df['min_flux']= df[['UV_filter', 'green_filter', 'red_filter', 'near_IR_filter', 'IR_filter']].min(axis=1)
    df['flux_range']= df['max_flux'] - df['min_flux']


    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    x_xgb_class=df.drop(['class', 'red_shift'], axis=1)
    y_class=df['class']

    le = LabelEncoder()
    y_class_encoded = le.fit_transform(y_class)

    split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(x_xgb_class, y_class_encoded):
        x_train_class=x_xgb_class.iloc[train_index]
        x_test_class=x_xgb_class.iloc[test_index]
        y_train_class=y_class_encoded[train_index]
        y_test_class=y_class_encoded[test_index]
    
    print(np.unique(y_train_class))
    print(le.classes_)

    classes=np.unique(y_train_class)
    weights=compute_class_weight(
        class_weight={0:1, 1:1, 2:2},
        classes=classes,
        y=y_train_class
    )

    class_weight_dict=dict(zip(classes, weights))

    sample_weight=np.array([class_weight_dict[i] for i in y_train_class])

    # xgb_class_model = XGBClassifier(
    #     n_estimators=300,
    #     learning_rate=0.05,
    #     max_depth=5,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     reg_lambda=1,
    #     reg_alpha=0.3,
    #     n_jobs=-1,
    #     random_state=42,
    #     use_label_encoder=False,
    #     eval_metric='mlogloss'
    # )

    xgb_class_model=XGBClassifier( #QSO focused model 
    n_estimators=800,
    max_depth=28,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1
    )
    xgb_class_model.fit(x_train_class, y_train_class, sample_weight=sample_weight)


    class_pred=xgb_class_model.predict(x_test_class)
    acc=accuracy_score(y_test_class, class_pred)
    print('XGBClassifier results:')
    print(f'Classification Accuracy: {acc}')
    print(classification_report(y_test_class, class_pred))
    print(f'Confusion Matrix:\n{confusion_matrix(y_test_class, class_pred)}')

    joblib.dump(xgb_class_model, 'models/XGBClassifier_model.pkl')
    joblib.dump(x_train_class.columns.tolist(), 'models/classifier_columns.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
def train_model():
    df=pd.read_csv('data/StarClassificationDataset.csv', low_memory=False)
    clean_data(df)
    df['alpha']= pd.to_numeric(df['alpha'], errors='coerce')
    df['alpha']=df['alpha'].fillna(df['alpha'].median())
    df=modify_df(df)

    df_notstar=df[df['class'] != 'STAR']
    
    x_train, x_test, y_train, y_test= split_data(df_notstar)

    y_train_log=np.log1p(y_train)
    y_test_log=np.log1p(y_test)

    #train model 
    # model=train_model(x_train, y_train)
    model= XGBRegressor(n_estimators=600, max_depth=6, objective='reg:squarederror', learning_rate= 0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=1, reg_alpha=0.3, n_jobs=-1, random_state=42)
    model.fit(x_train, y_train_log)

    joblib.dump(model, 'xgb_redshift_model.pkl')

    return model

def galaxy_model():
    df=pd.read_csv(FILE_PATH, low_memory=False)
    clean_data(df)
    df['alpha']= pd.to_numeric(df['alpha'], errors='coerce')
    df['alpha']=df['alpha'].fillna(df['alpha'].median())

    df=modify_df(df)

    df_galaxy=df[df['class'] == 'GALAXY']

    x_galaxy_train, x_galaxy_test, y_galaxy_train, y_galaxy_test= split_data(df_galaxy)
    y_galaxy_train_log=np.log1p(y_galaxy_train)

    #galaxy model
    galaxy_model=XGBRegressor(n_estimators=500, max_depth=6, objective='reg:squarederror', learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                            reg_lambda=1, reg_alpha=0.3, random_state=42)
    galaxy_model.fit(x_galaxy_train, y_galaxy_train_log)
    joblib.dump(galaxy_model, 'xgb_galaxy_redshift_model.pkl')



def quasar_model():
    df=pd.read_csv(FILE_PATH, low_memory=False)
    clean_data(df)
    df['alpha']= pd.to_numeric(df['alpha'], errors='coerce')
    df['alpha']=df['alpha'].fillna(df['alpha'].median())
    df=modify_df(df)   
    df_QSO=df[df['class']=='QSO']


    low_QSO=df_QSO[df_QSO['red_shift']< 1.25]
    mid_QSO=df_QSO[(df_QSO['red_shift']>=1.25) & (df_QSO['red_shift']<2)]
    high_QSO=df_QSO[df_QSO['red_shift']>=2]

    x_low_QSO=low_QSO.drop(['class', 'red_shift'], axis=1)
    y_low_QSO=low_QSO['red_shift']

    x_mid_QSO=mid_QSO.drop(['class', 'red_shift'], axis=1)
    y_mid_QSO=mid_QSO['red_shift']

    x_high_QSO=high_QSO.drop(['class', 'red_shift'], axis=1)
    y_high_QSO=high_QSO['red_shift']


    #train test split for each QSO bin
    x_low_QSO_train, x_low_QSO_test, y_low_QSO_train, y_low_QSO_test=split_data(low_QSO)

    x_mid_QSO_train, x_mid_QSO_test, y_mid_QSO_train, y_mid_QSO_test=split_data(mid_QSO)

    x_high_QSO_train, x_high_QSO_test, y_high_QSO_train, y_high_QSO_test=split_data(high_QSO)


    y_low_QSO_train_log=np.log1p(y_low_QSO_train)

    y_mid_QSO_train_log=np.log1p(y_mid_QSO_train)

    y_high_QSO_train_log=np.log1p(y_high_QSO_train)


    #QSO model

    #low QSO model
    low_qso_model=XGBRegressor(n_estimators=500, max_depth=6, objective='reg:squarederror', learning_rate=0.05, n_jobs=-1,
                            reg_lambda=1, reg_alpha=0.3, random_state=42)
    low_qso_model.fit(x_low_QSO_train, y_low_QSO_train_log)
    joblib.dump(low_qso_model, 'xgb_low_qso_redshift_model.pkl')


    #mid QSO model
    mid_qso_model=XGBRegressor(n_estimators=500, max_depth=6, objective='reg:squarederror', learning_rate=0.05, n_jobs=-1,
                            reg_lambda=1, reg_alpha=0.3, random_state=42)
    mid_qso_model.fit(x_mid_QSO_train, y_mid_QSO_train_log)
    joblib.dump(mid_qso_model, 'xgb_mid_qso_redshift_model.pkl')


    # high QSO model
    high_qso_model=XGBRegressor(n_estimators=500, max_depth=6, objective='reg:squarederror', learning_rate=0.05, n_jobs=-1,
                            reg_lambda=1, reg_alpha=0.3, random_state=42)
    high_qso_model.fit(x_high_QSO_train, y_high_QSO_train_log)
    joblib.dump(high_qso_model, 'xgb_high_qso_redshift_model.pkl')



def evaluate_model(model, x_test, y_test):
    y_pred=model.predict(x_test)
    mae=mean_absolute_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)
    print(f'mae: {mae}', f'r2: {r2}')


xgb_classifier_model()
# classifier_model()
# train_model()
# quasar_model()
# galaxy_model()