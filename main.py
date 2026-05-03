from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

# Load scaler only
scaler = joblib.load("scaler.pkl")

print("🚀 API Running Without TensorFlow")

# =========================
# LOAD DATA
# =========================
def get_user_data(username):

    df = pd.read_csv("expense_data.csv")

    df['uname'] = df['uname'].astype(str).str.strip()
    df = df[df['uname'] == username.strip()]

    if df.empty:
        return pd.DataFrame()

    df['product_price'] = pd.to_numeric(df['product_price'], errors='coerce')
    df['pdate'] = pd.to_datetime(df['pdate'], errors='coerce')

    df = df.dropna()

    daily = df.groupby('pdate', as_index=False)['product_price'].sum()

    return daily

# =========================
# SIMPLE TREND PREDICTION (NO AI MODEL)
# =========================
def predict_all(df):

    if df is None or df.empty:
        return {"error": "No data found"}

    values = df['product_price'].values.astype(float)

    if len(values) < 7:
        values = np.pad(values, (7 - len(values), 0), mode='edge')

    last7 = values[-7:]

    avg = np.mean(last7)
    trend = np.diff(last7).mean()

    next_day = avg + trend

    week = [next_day + i * trend for i in range(7)]
    month = [next_day + i * trend for i in range(30)]

    return {
        "next_day": round(float(next_day), 2),
        "weekly_total": round(float(np.sum(week)), 2),
        "monthly_total": round(float(np.sum(month)), 2),
        "weekly_list": [round(x, 2) for x in week]
    }

# =========================
# API
# =========================
@app.get("/predict")
def predict(username: str):

    data = get_user_data(username)

    return predict_all(data)
