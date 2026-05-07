🌌 Astronomical Object Classification & Distance Estimation

    A machine learning project that classifies celestial objects and estimates their distance from Earth using astrophysical principles.



# Overview

Stellar ML is an advanced machine learning project designed to classify astronomical objects based on photometric and observational data into three major categories:

* **GALAXY**
* **QSO (Quasar)**
* **STAR**

In addition to classification, the system also predicts:

* Estimated redshift
* Approximate cosmic distance
* Confidence score
* Scientific interpretation of results

This project combines astrophysics-inspired feature engineering with machine learning to simulate realistic astronomical object analysis.

---

# Key Features

## Object Classification

Uses machine learning models to classify objects into:

* Galaxy
* Quasar
* Star

---

## Redshift Prediction

Predicts cosmological redshift for:

* Galaxies
* QSOs

For stars:

* Redshift is expected near zero
* Local stellar interpretation is applied

---

## Distance Estimation

Converts redshift into approximate cosmic distance:

* Megaparsecs (MPC)
* Billion light-years

---

## Confidence Scoring

Provides prediction confidence percentages for reliability assessment.

---

# Machine Learning Models

## Primary Classifier

### XGBoost Classifier

* Main production model
* ~90% classification accuracy
* Strong GALAXY and QSO performance

---

## Redshift Model

### XGBoost Regressor

* Predicts redshift values
* Supports astrophysical distance estimation

---

# Feature Engineering

The project includes extensive domain-specific engineered features:

## Color Indices:

* u-g
* g-r
* r-i
* i-z

## Spectral Features:

* spectral_slope
* spectral_drop
* redness_index
* Blue_axis

## Flux Ratios:

* u_g_ratio
* g_r_ratio
* r_i_ratio
* i_z_ratio
* uv_ir_ratio

## Log Brightness:

* log_UV
* log_GREEN
* log_RED
* log_IR
* log_near_IR

## Shape & Curvature:

* color_curvature
* color_changes
* brightness_spread

## Flux Distribution:

* total_flux
* flux_mean
* flux_max
* flux_concentration
* uv_dominance
* spread_ratio
* flux_skew

## Advanced Features:

* spectral_entropy
* color_sharpness
* uv_peak_ratio
* red_tail_strength

---

# Current Performance

## Classification Accuracy:

### ~90.1%

### Strengths:

* GALAXY prediction: Excellent
* QSO prediction: Excellent
* STAR prediction: Moderate

---

# Known Limitations

## STAR Class:

* Lower recall than GALAXY/QSO
* Sometimes confused with:

  * QSO
  * GALAXY

### Future Improvements:

* Class balancing
* Threshold optimization
* STAR-specific correction rules
* Additional feature engineering

---

# Sample Output


🔭 Astronomical Analysis Result
Object Type: QSO
Confidence: 79.52%
Estimated Redshift: 0.85
Distance: 3621.77 MPC (~11.81 billion light-years)

Conclusion:
This object is an extremely distant quasar located in the far reaches of the universe.
```

---

# Project Structure

stellar_ml/
│
├── src/
│   ├── preprocess.py
│   ├── predict.py
│
├── models/
│   ├── XGBClassifier_model.pkl
│   ├── xgb_redshift_model.pkl
│   └── label_encoder.pkl
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

## Clone repository:

git clone https://github.com/Gauravvv15/Stellar_ML.git
cd stellar-ml
```

---

## Install dependencies:

pip install -r requirements.txt
```

---

# Usage

Run:

```
python main.py
```

Then input:

* alpha
* delta
* UV_filter
* green_filter
* red_filter
* near_IR_filter
* IR_filter
* MJD

---

# Future Roadmap

* Improve STAR recall
* Hyperparameter tuning
* Deploy Streamlit web app
* API integration
* Visualization dashboard
* Scientific reporting improvements

---

# Project Status

## Version: Stellar v1.0 Beta

### Development Stage:

Advanced portfolio-level ML system with ongoing optimization.

---

# Author

### Gauravv15

AIML Student | Machine Learning Enthusiast | Data Science Builder

---

# Final Note

Stellar ML demonstrates practical machine learning, astrophysical domain adaptation, feature engineering, and scientific prediction system design in a single integrated project.
