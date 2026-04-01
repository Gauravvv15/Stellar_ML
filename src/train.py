import joblib
import pandas as pd
from preprocess import modify_df, clean_data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from preprocess import clean_data

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

    class_model=RandomForestClassifier(n_estimators=150,
                                 max_depth=None, 
                                 random_state=42,
                                 class_weight='balanced')
    
    class_model.fit(x_train_class, y_train_class)

    joblib.dump(class_model, 'RandomForestClassifier_model.pkl')



def train_model():
    df=pd.read_csv('data/StarClassificationDataset.csv', low_memory=False)
    clean_data(df)
    df['alpha']= pd.to_numeric(df['alpha'], errors='coerce')
    df['alpha']=df['alpha'].fillna(df['alpha'].median())
    df=modify_df(df)

    x_train, x_test, y_train, y_test= split_data(df)

    #train model 
    model=train_model(x_train, y_train)
    model= XGBRegressor(n_estimators=100, max_depth=15, subsample=0.8, colsample_bytree=0.8, learning_rate=0.1, random_state=42)
    model.fit(x_train, y_train)

    joblib.dump(model, 'xgb_redshift_model.pkl')

    return model


def evaluate_model(model, x_test, y_test):
    y_pred=model.predict(x_test)
    mae=mean_absolute_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)
    print(f'mae: {mae}', f'r2: {r2}')

classifier_model()