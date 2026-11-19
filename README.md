# 🐾 Possum Total Length Predictor

A machine learning project on the **possum dataset** (104 mountain brushtail possums from 7 sites in
Australia). It explores the data, handles missing values, and trains a **linear regression** model that
predicts a possum's **total body length** (`totlngth`, cm) from its other body measurements. An
interactive **Streamlit** app lets you make predictions and inspect the model.

## 🚀 Features

- Data exploration and missing-value handling (`age` and `footlgth` have 3 missing values in total)
- Preprocessing demo: KNN imputation, ordinal encoding and standard scaling (`encoding_imputer_Scaling.ipynb`)
- Linear regression for total length, evaluated with R², MAE, MSE and RMSE (`linearRegression.ipynb`)
- **Streamlit demo** with:
  - Single prediction from 10 body measurements, with a training-range check
  - Batch prediction from an uploaded CSV, with a downloadable result
  - Model performance: train/test metrics, 5-fold cross-validated R², actual-vs-predicted plot, coefficients
  - Dataset view with feature correlations

## 📊 Model

| | |
|---|---|
| Target | `totlngth` – total length (cm) |
| Features | `site`, `age`, `hdlngth`, `skullw`, `taill`, `footlgth`, `earconch`, `eye`, `chest`, `belly` |
| Excluded | `sex`, `Pop` (as in the notebook), `case` (row ID) |
| Preprocessing | Mean imputation of missing values |
| Algorithm | scikit-learn `LinearRegression` |
| Split | 80/20, `random_state=42` |

Current results: train R² ≈ 0.80, test R² ≈ 0.41, test MAE ≈ 1.8 cm, shuffled 5-fold CV R² ≈ 0.68 ± 0.14.
The test set has only 21 rows, so the cross-validated score is the more reliable estimate.

## 🛠 Tech Stack

Python · pandas · NumPy · scikit-learn · Altair · Streamlit · Jupyter

## 📂 Project Structure

```
possum/
├── possum.csv                       # Dataset
├── linearRegression.ipynb           # EDA + linear regression notebook
├── encoding_imputer_Scaling.ipynb   # Imputation / encoding / scaling notebook
├── possum_model.py                  # Reusable training, evaluation and prediction code
├── app.py                           # Streamlit demo
├── requirements.txt
└── .streamlit/config.toml
```

## 📦 Installation

```bash
git clone https://github.com/amirafarhan/possum.git
cd possum
python -m venv venv
# Windows: venv\Scripts\activate    Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Run

Streamlit demo (opens http://localhost:8501):

```bash
streamlit run app.py
```

Train and print metrics from the command line:

```bash
python possum_model.py
```

Notebooks: `pip install jupyter`, run `jupyter notebook`, then open `linearRegression.ipynb` or
`encoding_imputer_Scaling.ipynb`.

## 🔮 Possible Next Steps

- Compare with regularised models (Ridge/Lasso) using cross-validation
- Treat `site` as a categorical feature (one-hot encoding)
- Add `sex` and `Pop` as encoded features

## 👨‍💻 Author

Amira Mohamed – Machine Learning Engineer
