from fastapi import FastAPI
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

app = FastAPI()

# =========================
# LOAD MODEL + SCALER
# =========================
model = tf.keras.models.load_model("lstm_model.keras", compile=False)
scaler = joblib.load("scaler.pkl")

print("🚀 Model Loaded Successfully")

# =========================
# LOAD DATA
# =========================
def get_user_data(username):

    df = pd.read_csv("expense_data.csv")

    # clean username
    df['uname'] = df['uname'].astype(str).str.strip()
    df = df[df['uname'] == username.strip()]

    if df.empty:
        return pd.DataFrame()

    # clean numeric
    df['product_price'] = pd.to_numeric(df['product_price'], errors='coerce')

    # clean date
    df['pdate'] = pd.to_datetime(df['pdate'], errors='coerce')

    df = df.dropna()

    # daily grouping
    daily = df.groupby('pdate', as_index=False)['product_price'].sum()

    daily = daily.sort_values('pdate')

    return daily

# =========================
# SAFE PREDICTION ENGINE
# =========================
def predict_days(base_scaled, days):

    temp = base_scaled.copy()
    preds = []

    for _ in range(days):

        # FORCE shape safety
        if len(temp) < 7:
            temp = np.vstack([temp, temp[-1]])  # repeat last value safely

        x = temp[-7:].reshape(1, 7, 1)

        pred = model.predict(x, verbose=0)[0][0]

        preds.append(pred)

        temp = np.vstack([temp, [[pred]]])

    return scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()

# =========================
# MAIN LOGIC
# =========================
def predict_all(df):

    if df is None or df.empty:
        return {"error": "No data found for user"}

    values = df['product_price'].astype(float).values

    # ensure numeric safety
    values = np.nan_to_num(values)

    # FORCE minimum 7 values
    if len(values) < 7:
        values = np.pad(values, (7 - len(values), 0), mode='edge')

    last7 = values[-7:].reshape(-1, 1)

    # scaler safety
    try:
        scaled = scaler.transform(last7)
    except Exception as e:
        return {"error": f"Scaler error: {str(e)}"}

    week = predict_days(scaled, 7)
    month = predict_days(scaled, 30)

    return {
        "next_day": round(float(week[0]), 2),
        "weekly_total": round(float(np.sum(week)), 2),
        "monthly_total": round(float(np.sum(month)), 2),
        "weekly_list": [round(float(x), 2) for x in week]
    }

# =========================
# API ENDPOINT
# =========================
@app.get("/predict")
def predict(username: str):

    try:
        print("\n🔥 API CALL:", username)

        data = get_user_data(username)

        print("📊 ROWS:", len(data))

        result = predict_all(data)

        print("✅ RESULT:", result)

        return result

    except Exception as e:
        import traceback

        print("\n❌ ERROR TRACEBACK:")
        traceback.print_exc()

        return {"error": str(e)}