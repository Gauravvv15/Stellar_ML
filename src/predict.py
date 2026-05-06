import joblib
from matplotlib.pyplot import rc
import pandas as pd
import numpy as np
from src.astronomical_calcualtions import redshift_to_mpc, mpc_to_lightyears

def class_predict(x):
    xgb_model=joblib.load('models/XGBClassifier_model.pkl')
    le=joblib.load('models/label_encoder.pkl')
    cols=joblib.load('models/classifier_columns.pkl')
    x=x.reindex(columns=cols, fill_value=0)


    xgb_pred=xgb_model.predict_proba(x)[0]

    final_index=np.argmax(xgb_pred)
    final_confidence=xgb_pred[final_index]

    final_class=le.inverse_transform([final_index])[0]
    

    if final_confidence > 0.60:

        star_like=(
        x['u-g'].iloc[0] > 1.0 and
        x['color_changes'].iloc[0] > 0.50 and
        x['color_curvature'].iloc[0] > 0.6 and
        x['brightness_spread'].iloc[0] > 0.65 
        )
        
        if final_class=="QSO" and final_confidence > 0.95:
            if star_like:
                final_class='STAR'
                final_confidence=max(0.75, final_confidence*0.9)
                model_used="Brightness Rule Override(Bright in UV and Green filters suggests STAR.)"


            else:
                 model_used = "XGB Primary Classifier"

        else:
            model_used = "XGB Primary Classifier"

    else: 
        model_used = "XGB Primary Classifier (Low Confidence)"

    #     elif rf_class=="STAR" and rf_confidence > 0.40:
    #         final_class="STAR"
    #         final_confidence=rf_confidence
    #         model_used="Random Forest Classifier (STAR specialist override)"


    #     elif rf_class in  ['STAR', 'GALAXY']:
    #         final_class=rf_class
    #         final_confidence=rf_confidence
    #         model_used="Random Forest Classifier (STAR/GALAXY specialist)"


    #     else: #fallback    
    #         if rf_confidence > xgb_confidence:
    #             final_class=rf_class
    #             final_confidence=rf_confidence
    #             model_used="Random Forest Classifier (Fallback)"

    #         else:
    #             final_class=xgb_class
    #             final_confidence=xgb_confidence
    #             model_used="XGBoost Classifier (Fallback)"
        
    # else:
    #     final_class=rf_class
    #     final_confidence=confidance
    #     model_used="Random Forest Classifier (Uncertain)"
        
    #     # class_pred=le.inverse_transform([np.argmax([rf_pred[0], xgb_pred[0]])])[0]

    
    if final_class=="STAR":
        return(
            "🔭 Astronomical Analysis result\n"
            f"object type: {final_class}\n"
            f"Confidence: {final_confidence*100:.2f}%\n"
            f"Model Used: {model_used}\n"
            "Interpretation: This object is classified as a STAR, located within our galaxy (Milky Way).\n\n"
            "conclusion:\n This object is relatively close to us in cosmic terms, likely within a few thousand Light-years and it does not require cosmological distance Estimation."
        )
        
    interpretation=""
    if final_class=="GALAXY":
        interpretation="A GALAXY is a massive system of stars, interstellar gas, dust, and dark matter located outside of our galaxy(Milky Way).The distance to a galaxy can vary widely, from a few million light-years to billions of light-years, depending on its location in the universe."
        model_used=model_used
        dist=galaxy_model(x)

    elif final_class=="QSO":
        interpretation="A QUASAR (Quasi-Stellar Object) is an extremely luminous active galactic nucleus, powered by a supermassive black hole at its center. Quasars are among the most distant and energetic objects in the universe, often located billions of Light-years away from Earth."
        model_used=model_used
        dist=quasar_model(x)
        
    return(
        "🔭 Astronomical Analysis result\n"
        f"object type: {final_class}\n"
        f"Confidence: {final_confidence*100:.2f}%\n"
        f"Model Used: {model_used}\n"
        f"Interpretation: {interpretation}\n\n"
        f"Estimated Redshift: {dist['redshift']:.2f}\n"
        f"Distance: {dist['mpc']:.2f} MPC (~ {dist['lightyears']/1e9:.2f} billion light-years away)\n\n"
        "conclusion:\n"
        f"This object is extremely distant {final_class.lower()} located in the far reaches of the universe, and its light has taken billions of years to reach us."                
    )

def galaxy_model(x):
    model=joblib.load('models/xgb_galaxy_redshift_model.pkl')
    cols = joblib.load('models/training_columns.pkl')
    x=x.select_dtypes(include=[np.number])  #select only numeric features for quasar model
    x = x.reindex(columns=cols, fill_value=0)
    y_pred_log=model.predict(x)
    y_pred=np.expm1(y_pred_log)

    mpc=redshift_to_mpc(y_pred)
    lightyears=mpc_to_lightyears(mpc)

    return{
        "redshift": y_pred.item(),
        "mpc": mpc.item(),
        "lightyears":lightyears.item()
    }

def quasar_model(x):
    cols = joblib.load('models/training_columns.pkl')
    x=x.select_dtypes(include=[np.number])  #select only numeric features for quasar model
    x = x.reindex(columns=cols, fill_value=0)  
    low_qso_model=joblib.load('models/xgb_low_qso_redshift_model.pkl')
    mid_qso_model=joblib.load('models/xgb_mid_qso_redshift_model.pkl')
    high_qso_model=joblib.load('models/xgb_high_qso_redshift_model.pkl')

    # y=redshift_predict(x).item()
    low_pred=np.expm1(low_qso_model.predict(x))
    mid_pred=np.expm1(mid_qso_model.predict(x))
    high_pred=np.expm1(high_qso_model.predict(x))

    if low_pred < 1.25:
        model=low_qso_model
    
    elif 1.25 <= mid_pred < 2:
        model=mid_qso_model
    
    elif high_pred >= 2:
        model=high_qso_model

    y_pred_log=model.predict(x)
    y_pred=np.expm1(y_pred_log)

    mpc=redshift_to_mpc(y_pred)
    lightyears=mpc_to_lightyears(mpc)

    return{
        "redshift": y_pred.item(),
        "mpc": mpc.item(),
        "lightyears":lightyears.item()
    }

def redshift_predict(x):

    model=joblib.load('models/xgb_redshift_model.pkl')

    y_pred_log=model.predict(x)
    y_pred=np.expm1(y_pred_log)
    
    # mpc=redshift_to_mpc(y_pred)         #millions of parsecs
    # lightyears=mpc_to_lightyears(mpc)   #persecs is a unit of distance used in astronomy, 1 persec is about 3.26 light_years

    return y_pred

    # return {"redshift": y_pred.item(),
    # "mpc": mpc.item(),
    # "lightyears": lightyears.item()
    # }