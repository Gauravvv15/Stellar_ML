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
from sklearn.preprocessing import LabelEncoder

BASE_DIR=os.path.dirname(os.path.dirname(__file__))
FILE_PATH=os.path.join(BASE_DIR, 'data','StarClassificationDataset.csv')


def split_data(data):
    x=data.drop(['class', 'red_shift'], axis=1)
    y=data['red_shift']
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)
    joblib.dump(x_train.columns.tolist(), 'models/classifier_columns.pkl')
    return x_train, x_test, y_train, y_test


def xgb_classifier_model():
    df = pd.read_csv('data/StarClassificationDataset.csv', low_memory=False)
    clean_data(df)
    df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')
    df['alpha'] = df['alpha'].fillna(df['alpha'].median())
    df = modify_df(df)


    le = LabelEncoder()
    y_all = le.fit_transform(df['class'])
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    x_all = df.drop(['class', 'red_shift'], axis=1)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(x_all, y_all):
        x_train = x_all.iloc[train_index]
        x_test  = x_all.iloc[test_index]
        y_train = y_all[train_index]
        y_test  = y_all[test_index]


    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',  
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))


    star_idx = list(le.classes_).index('STAR')
    class_weight_dict[star_idx] *= 1.5
    print(f"Class weights: {dict(zip(le.classes_, class_weight_dict.values()))}")

    sample_weight = np.array([class_weight_dict[i] for i in y_train])


    xgb_class_model = XGBClassifier(
        n_estimators=500,
        max_depth=7,           
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=0.2,
        reg_lambda=1.5,
        min_child_weight=5,    
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )

    xgb_class_model.fit(
        x_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(x_test, y_test)],
        verbose=50
    )

    class_pred = xgb_class_model.predict(x_test)
    acc = accuracy_score(y_test, class_pred)

    print('\n' + '='*50)
    print('XGBClassifier Results:')
    print(f'Accuracy: {acc:.4f}')
    print(classification_report(y_test, class_pred, target_names=le.classes_))
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, class_pred)}')
    print('Rows=Actual, Cols=Predicted | Order:', list(le.classes_))
    print('='*50)

    joblib.dump(xgb_class_model, 'models/XGBClassifier_model.pkl')
    joblib.dump(x_train.columns.tolist(), 'models/classifier_columns.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')

    return xgb_class_model, le


def galaxy_model():
    df = pd.read_csv(FILE_PATH, low_memory=False)
    clean_data(df)
    df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')
    df['alpha'] = df['alpha'].fillna(df['alpha'].median())
    df = modify_df(df)

    df_galaxy = df[df['class'] == 'GALAXY']
    x_train, x_test, y_train, y_test = split_data(df_galaxy)
    y_train_log = np.log1p(y_train)

    gal_model = XGBRegressor(
        n_estimators=500, max_depth=6,
        objective='reg:squarederror',
        learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, reg_lambda=1, reg_alpha=0.3, random_state=42
    )
    gal_model.fit(x_train, y_train_log)

    y_pred = np.expm1(gal_model.predict(x_test))
    print(f"Galaxy model — MAE: {mean_absolute_error(y_test, y_pred):.4f}, R²: {r2_score(y_test, y_pred):.4f}")
    joblib.dump(gal_model, 'models/xgb_galaxy_redshift_model.pkl')


def quasar_model():
    df = pd.read_csv(FILE_PATH, low_memory=False)
    clean_data(df)
    df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')
    df['alpha'] = df['alpha'].fillna(df['alpha'].median())
    df = modify_df(df)

    df_QSO = df[df['class'] == 'QSO']

    low_QSO  = df_QSO[df_QSO['red_shift'] < 1.25]
    mid_QSO  = df_QSO[(df_QSO['red_shift'] >= 1.25) & (df_QSO['red_shift'] < 2)]
    high_QSO = df_QSO[df_QSO['red_shift'] >= 2]

    for name, subset, fname in [
        ('Low QSO',  low_QSO,  'models/xgb_low_qso_redshift_model.pkl'),
        ('Mid QSO',  mid_QSO,  'models/xgb_mid_qso_redshift_model.pkl'),
        ('High QSO', high_QSO, 'models/xgb_high_qso_redshift_model.pkl'),
    ]:
        x_tr, x_te, y_tr, y_te = split_data(subset)
        y_tr_log = np.log1p(y_tr)
        m = XGBRegressor(
            n_estimators=500, max_depth=6,
            objective='reg:squarederror',
            learning_rate=0.05, n_jobs=-1,
            reg_lambda=1, reg_alpha=0.3, random_state=42
        )
        m.fit(x_tr, y_tr_log)
        y_pred = np.expm1(m.predict(x_te))
        print(f"{name} — MAE: {mean_absolute_error(y_te, y_pred):.4f}, R²: {r2_score(y_te, y_pred):.4f}")
        joblib.dump(m, fname)



print("Training XGB Classifier...")
xgb_classifier_model()

print("Training Galaxy redshift model...")
galaxy_model()
print("Training QSO redshift models...")
quasar_model()
