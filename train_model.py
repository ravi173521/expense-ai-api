import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler

print("\n🚀 TRAINING STARTED...\n")

# =========================
# LOAD CSV
# =========================
df = pd.read_csv("expense_data.csv")

# clean data
df['product_price'] = pd.to_numeric(df['product_price'], errors='coerce')
df['pdate'] = pd.to_datetime(df['pdate'], errors='coerce')
df['uname'] = df['uname'].astype(str).str.strip()

df = df.dropna()

print("📊 Raw rows:", len(df))

# =========================
# DAILY GROUPING
# =========================
daily = df.groupby('pdate', as_index=False)['product_price'].sum()
daily = daily.sort_values('pdate')

print("📊 Daily rows:", len(daily))

if len(daily) < 8:
    raise ValueError("❌ Need at least 8 days of data")

# =========================
# SCALE DATA
# =========================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(daily[['product_price']])

joblib.dump(scaler, "scaler.pkl")

print("✅ Scaler saved")

# =========================
# SEQUENCES (7 DAYS)
# =========================
X, y = [], []

for i in range(7, len(scaled)):
    X.append(scaled[i-7:i])
    y.append(scaled[i])

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], X.shape[1], 1)

print("📊 Training shape:", X.shape)

# =========================
# MODEL
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(7, 1)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# =========================
# TRAIN
# =========================
model.fit(X, y, epochs=20, verbose=1)

# =========================
# SAVE MODEL
# =========================
model.save("lstm_model.keras")

print("\n🎯 TRAINING COMPLETE")