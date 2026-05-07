import joblib
import pandas as pd
import numpy as np
from src.astronomical_calcualtions import redshift_to_mpc, mpc_to_lightyears



def _is_star_like(x):
    """
    Returns (is_star_like: bool, star_score: float 0–1)
    star_score aggregates how many star-like conditions are met.
    """
    score = 0.0
    checks = 0


    ug = x['u-g'].iloc[0]
    if ug > 1.0:
        score += 1
    checks += 1


    if 'colour_symmetry' in x.columns:
        cs = x['colour_symmetry'].iloc[0]
        if cs < 1.0:
            score += 1
        checks += 1


    if 'sed_uniformity' in x.columns:
        su = x['sed_uniformity'].iloc[0]
        if su < 0.05:
            score += 1
        checks += 1

 
    if 'blue_red_gradient' in x.columns:
        brg = x['blue_red_gradient'].iloc[0]
        if abs(brg) < 0.05:
            score += 1
        checks += 1

    
    ri = x['redness_index'].iloc[0]
    if abs(ri) < 2.0:
        score += 1
    checks += 1

    bs = x['brightness_spread'].iloc[0]
    if bs < 1.5:
        score += 1
    checks += 1

    star_score = score / checks if checks > 0 else 0
    is_star = star_score >= 0.70  # majority of conditions met
    return is_star, star_score


def class_predict(x):
    xgb_model = joblib.load('models/XGBClassifier_model.pkl')
    le         = joblib.load('models/label_encoder.pkl')
    cols       = joblib.load('models/classifier_columns.pkl')
    x = x.reindex(columns=cols, fill_value=0)

    xgb_proba = xgb_model.predict_proba(x)[0]

    final_index      = np.argmax(xgb_proba)
    final_confidence = xgb_proba[final_index]
    final_class      = le.inverse_transform([final_index])[0]

    star_idx    = list(le.classes_).index('STAR')
    star_proba  = xgb_proba[star_idx]

    model_used = "XGB Primary Classifier"



    if final_confidence < 0.80:
        is_star, star_score = _is_star_like(x)
        if is_star and star_proba > 0.25:

            final_class      = 'STAR'
            final_confidence = max(star_proba, 0.60)
            model_used       = f"STAR Photometric Override (score={star_score:.2f}, low confidence region)"


    if final_class == 'GALAXY' and final_confidence < 0.92:
        is_star, star_score = _is_star_like(x)
        if is_star and star_proba > 0.30:
            final_class      = 'STAR'
            final_confidence = max(star_proba, 0.60)
            model_used       = f"STAR Override from GALAXY (score={star_score:.2f})"



    if final_class == 'QSO' and final_confidence < 0.95:
        is_star, star_score = _is_star_like(x)
        if is_star and star_proba > 0.25:
            final_class      = 'STAR'
            final_confidence = max(star_proba, 0.60)
            model_used       = f"STAR Override from QSO (score={star_score:.2f})"



    if final_class == "STAR":
        return (
            "🔭 Astronomical Analysis Result\n"
            f"  Object Type  : {final_class}\n"
            f"  Confidence   : {final_confidence * 100:.2f}%\n"
            f"  Model Used   : {model_used}\n\n"
            "  Interpretation:\n"
            "  This object is classified as a STAR — a local Milky Way object.\n"
            "  Stars have near-zero cosmological redshift (z ≈ 0.00–0.004),\n"
            "  so cosmological distance estimation does not apply here.\n\n"
            "  Conclusion:\n"
            "  This object is likely within a few thousand light-years of Earth,\n"
            "  well within our own galaxy. No redshift distance model applied."
        )

    interpretation = ""
    dist = None

    if final_class == "GALAXY":
        interpretation = (
            "A GALAXY is a massive system of stars, gas, dust, and dark matter "
            "located outside the Milky Way. Distances range from a few million "
            "to billions of light-years depending on its redshift."
        )
        dist = galaxy_model(x)

    elif final_class == "QSO":
        interpretation = (
            "A QUASAR (Quasi-Stellar Object) is an extremely luminous active "
            "galactic nucleus powered by a supermassive black hole. Quasars are "
            "among the most distant and energetic objects in the universe."
        )
        dist = quasar_model(x)

    return (
        "🔭 Astronomical Analysis Result\n"
        f"  Object Type  : {final_class}\n"
        f"  Confidence   : {final_confidence * 100:.2f}%\n"
        f"  Model Used   : {model_used}\n\n"
        f"  Interpretation:\n  {interpretation}\n\n"
        f"  Estimated Redshift : {dist['redshift']:.4f}\n"
        f"  Distance           : {dist['mpc']:.2f} Mpc "
        f"(~{dist['lightyears'] / 1e9:.2f} billion light-years)\n\n"
        "  Conclusion:\n"
        f"  This is a distant {final_class.lower()} whose light has travelled "
        f"billions of years to reach us."
    )



def galaxy_model(x):
    model = joblib.load('models/xgb_galaxy_redshift_model.pkl')
    cols  = joblib.load('models/classifier_columns.pkl')
    x = x.select_dtypes(include=[np.number])
    x = x.reindex(columns=cols, fill_value=0)

    y_pred = np.expm1(model.predict(x))
    mpc        = redshift_to_mpc(y_pred)
    lightyears  = mpc_to_lightyears(mpc)
    return {"redshift": y_pred.item(), "mpc": mpc.item(), "lightyears": lightyears.item()}


def quasar_model(x):
    cols = joblib.load('models/classifier_columns.pkl')
    x = x.select_dtypes(include=[np.number])
    x = x.reindex(columns=cols, fill_value=0)

    low_model  = joblib.load('models/xgb_low_qso_redshift_model.pkl')
    mid_model  = joblib.load('models/xgb_mid_qso_redshift_model.pkl')
    high_model = joblib.load('models/xgb_high_qso_redshift_model.pkl')

    low_pred  = np.expm1(low_model.predict(x)).item()
    mid_pred  = np.expm1(mid_model.predict(x)).item()
    high_pred = np.expm1(high_model.predict(x)).item()

    # Pick model based on which bin the predictions fall into
    if low_pred < 1.25:
        chosen = low_model
    elif 1.25 <= mid_pred < 2.0:
        chosen = mid_model
    else:
        chosen = high_model

    y_pred     = np.expm1(chosen.predict(x))
    mpc        = redshift_to_mpc(y_pred)
    lightyears  = mpc_to_lightyears(mpc)
    return {"redshift": y_pred.item(), "mpc": mpc.item(), "lightyears": lightyears.item()}


def redshift_predict(x):
    model  = joblib.load('models/xgb_redshift_model.pkl')
    y_pred = np.expm1(model.predict(x))
    return y_pred