🌌 Astronomical Object Classification & Distance Estimation

    A machine learning project that classifies celestial objects and estimates their distance from Earth using astrophysical principles.

~   Overview

    This project combines machine learning with astrophysics to classify celestial objects and estimate their cosmological distance.
    This project uses data from the Sloan Digital Sky Survey (SDSS) to:

🔭 Classify astronomical objects into:
    STAR
    GALAXY
    QSO (Quasar)

    🌌 Predict redshift (z) using regression models
    📏 Convert redshift into distance (Megaparsecs & Light-years) using cosmological formulas

Key Features:
    Advanced feature engineering:
    Color indices (u-g, g-r, r-i, i-z)
    Flux ratios
    Log transformations
    Normalized spectral features

Machine Learning Models:
    Classification: Random Forest
    Regression: XGBoost

Scientific computation:
    Redshift → Distance conversion using Hubble’s Law

End-to-end pipeline:
    Input → Feature Engineering → Prediction → Distance Estimation

📊 Results
    Task	Performance
    Multiclass Classification	~98–99% Accuracy
    Redshift Regression	R² ≈ 0.96

🌌 Distance Calculation

    Distance is calculated using:

    𝑑 =z.c/H0

    Where:
    z = redshift
    c = speed of light (300,000 km/s)
    H0= Hubble constant (~70 km/s/Mpc)

Then converted to light-years:
    1 Mpc ≈ 3.26 million light-years

📁 Project Structure
    galaxy_project/
    │
    ├── data/
    │   └── StarClassificationDataset.csv
    │
    ├── models/
    │   ├── model.pkl
    │   └── columns.pkl
    │
    ├── src/
    │   ├── preprocess.py
    │   ├── train.py
    │   ├── predict.py
    │   └── utils.py
    │
    ├── notebooks/
    │   └── experiments.ipynb
    │
    ├── main.py
    ├── requirements.txt
    └── README.md

⚙️ Installation
    git clone https://github.com/your-username/galaxy-project.git
    cd galaxy-project
    pip install -r requirements.txt
    ▶️ Usage
    1. Train the model
    python src/train.py
    2. Run prediction
    python main.py

Enter values like:
    alpha, delta, UV, green, red, near_IR, IR, MJD

🧪 Example Output
    The object is approximately 3760.48 Mpc away,
    which is about 12.26 billion light-years.

⚠️ Challenges & Learnings
    🔴 Data Leakage (Major Issue)
    Initially included derived features using redshift
    Resulted in unrealistic near-perfect accuracy
    Fixed by removing leakage features
    🔴 Feature Mismatch
    Training and prediction inputs had different structures
    Fixed using column alignment (reindex)
    🔴 Unit Conversion Error
    Incorrect conversion from Mpc to light-years
    Corrected using proper astrophysical constants

🧠 Key Learnings
    Importance of feature consistency
    Detecting and fixing data leakage
    Handling real-world ML pipeline issues
    Combining domain knowledge + ML

🚀 Future Improvements
    Add uncertainty estimation (confidence intervals)
    Improve high redshift predictions
    Deploy using Streamlit / Flask
    Add real-time API support

🛠️ Tech Stack
    Python
    Pandas, NumPy
    Scikit-learn
    XGBoost
    Matplotlib
    Joblib


📌 Author
    Gaurav
    Aspiring Data Scientist (AI/ML)

    ⭐ If you like this project