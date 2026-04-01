import joblib
import pandas as pd
from src.astronomical_calcualtions import redshift_to_mpc, mpc_to_lightyears

def class_predict(x):
    class_model=joblib.load('models/RandomForestClassifier_model.pkl')

    class_pred=class_model.predict(x)[0]

    if class_pred=="STAR":
        # return f"The class of this object: {class_pred}."
        return(
            "🔭 Astronomical Analysis result\n"
            f"object type: {class_pred}\n"
            "Interpretation: This object is classified as a STAR, located within our galaxy (Milky Way).\n\n"
            "conclusion:\n This object is relatively close to us in cosmic terms, likely within a few thousand Light-years and it does not require cosmological distance Estimation."
        )
    #for galaxy and Quasar
    dist=distance_predict(x)

    interpretation=""
    if class_pred=="GALAXY":
        interpretation="A GALAXY is a massive system of stars, interstellar gas, dust, and dark matter located outside of our galaxy(Milky Way).The distance to a galaxy can vary widely, from a few million light-years to billions of light-years, depending on its location in the universe."

    elif class_pred=="QSO":
        interpretation="A QUASAR (Quasi-Stellar Object) is an extremely luminous active galactic nucleus, powered by a supermassive black hole at its center. Quasars are among the most distant and energetic objects in the universe, often located billions of Light-years away from Earth."

    return(
        "🔭 Astronomical Analysis result\n"
        f"object type: {class_pred}\n"
        f"Interpretation: {interpretation}\n\n"
        f"Estimated Redshift: {dist['redshift']:.2f}\n"
        f"Distance: {dist['mpc']:.2f} MPC (~ {dist['lightyears']/1e9:.2f} billion light-years away)\n\n"
        "conclusion:\n"
        f"This object is extremely distant {class_pred.lower()} located in the far reaches of the universe, and its light has taken billions of years to reach us."
                        
    )


def distance_predict(x):

    model=joblib.load('models/xgb_redshift_model.pkl')

    y_pred=model.predict(x)
    
    mpc=redshift_to_mpc(y_pred)         #millions of parsecs
    lightyears=mpc_to_lightyears(mpc)   #persecs is a unit of distance used in astronomy, 1 persec is about 3.26 light_years

    # print(f'DEBUG: z={y_pred}, mpc={mpc}, lightyears={lightyears}')
    # return f"The object is approximately {mpc.item():.2f} megaparsecs(mpc) away, which is about {lightyears.item()/1e9:.2f} billion of light-years."

    # print(f'DEBUG: z={y_pred}, mpc={mpc}, lightyears={lightyears}')
    return {"redshift": y_pred.item(),
    "mpc": mpc.item(),
    "lightyears": lightyears.item()
    }