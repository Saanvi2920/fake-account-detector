# Instagram Fake Account Detection & Auto-Reporting System

A machine learning system that automatically detects and reports fake/spam Instagram accounts using a stack ensemble of 5 ML models.

## Features
- **Single Account Detection**: Enter account features to predict if it's fake
- **Bulk Detection**: Scan thousands of accounts at once
- **Auto-Reporting**: Automatically generates reports with risk levels and reasons
- **Dashboard**: View all reports filtered by risk level
- **Stack Ensemble**: Combines LR, Decision Tree, Random Forest, XGBoost, and ANN

## How to Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 in your browser.

## Deploy to Render (Free)
1. Push this project to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and deploys
5. Your app will be live at `https://your-app.onrender.com`

## Project Structure
```
app.py                 - Flask web application
auto_report.py         - Auto-reporting system with SQLite
train_models.py        - Train the 5 base ML models
stack_and_evaluate.py  - Stack ensemble + evaluation + SHAP
models/                - Saved trained models (.joblib)
data/                  - ML-ready datasets
reports/               - Auto-generated report JSONs + SQLite DB
evaluation/            - Confusion matrix, ROC curve, SHAP plots
```

## Tech Stack
- Python, Flask, scikit-learn, XGBoost, SHAP, SQLite
- Hosted on Render
