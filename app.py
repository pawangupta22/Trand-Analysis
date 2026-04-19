import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.title("Trend Analysis Dashboard")

model = joblib.load("model.pkl")
le = joblib.load("encoder.pkl")

file = st.file_uploader("Upload Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    # =========================
    # CLEANING
    # =========================
    df['date'] = pd.to_datetime(df[['Year','Month','Day','Hour']], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    df['engagement'] = df['Likes'] + df['Retweets']

    # =========================
    # DATASET OVERVIEW
    # =========================
    st.subheader("Overview")
    st.write({
        "Total Rows": len(df),
        "Unique Hashtags": df['Hashtags'].nunique(),
        "Avg Engagement": int(df['engagement'].mean())
    })

    # =========================
    # SPLIT INTO 4 PARTS
    # =========================
    df = df.reset_index(drop=True)

    n = len(df)

    split1 = int(n * 0.25)
    split2 = int(n * 0.50)
    split3 = int(n * 0.75)

    part1 = df.iloc[:split1]
    part2 = df.iloc[split1:split2]
    part3 = df.iloc[split2:split3]
    part4 = df.iloc[split3:]

    old_data = df.iloc[:split3]
    latest_data = df.iloc[split3:]
    # =========================
    # CURRENT TREND
    # =========================
    st.subheader("Current Trends")

    current = latest_data.groupby('Hashtags')['engagement'].mean().sort_values(ascending=False)
    st.dataframe(current.head(5))

    # =========================
    # TREND DIRECTION
    # =========================
    old_avg = old_data.groupby('Hashtags')['engagement'].mean()
    latest_avg = latest_data.groupby('Hashtags')['engagement'].mean()

    trend = pd.DataFrame({
        "old": old_avg,
        "latest": latest_avg
    }).fillna(0)

    trend["change_pct"] = ((trend["latest"] - trend["old"]) / (trend["old"] + 1)) * 100

    rising = trend[trend["change_pct"] > 3]
    falling = trend[trend["change_pct"] < -3]

    st.subheader("Rising Trends")
    st.dataframe(rising.sort_values("change_pct", ascending=False).head(5))

    st.subheader("Falling Trends")
    st.dataframe(falling.sort_values("change_pct").head(5))

    # =========================
    # FEATURES
    # =========================
    df['day_number'] = (df['date'] - df['date'].min()).dt.days
    df['ma_3'] = df['engagement'].rolling(3).mean().fillna(0)
    df['ma_7'] = df['engagement'].rolling(7).mean().fillna(0)
    df['growth_rate'] = df['engagement'].pct_change().fillna(0).clip(-5, 5)

    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['hour'] = df['date'].dt.hour

    df['hashtag_encoded'] = le.transform(df['Hashtags'])

    # =========================
    # FUTURE PREDICTION
    # =========================
    last_day = df['day_number'].max()

    future = pd.DataFrame({
        'day_number': np.arange(last_day+1, last_day+8),
        'ma_3': [df['ma_3'].iloc[-1]]*7,
        'ma_7': [df['ma_7'].iloc[-1]]*7,
        'growth_rate': [df['growth_rate'].iloc[-1]]*7,
        'day_of_week': [df['day_of_week'].iloc[-1]]*7,
        'month': [df['month'].iloc[-1]]*7,
        'hour': [df['hour'].iloc[-1]]*7,
        'hashtag_encoded': [df['hashtag_encoded'].iloc[-1]]*7
    })

    preds = model.predict(future)

    st.subheader("🔮 Next 7 Days Prediction")
    st.write(preds.astype(int))

    # =========================
    # GRAPH
    # =========================
    st.subheader("Engagement Over Time")

    fig, ax = plt.subplots()
    ax.plot(df['date'], df['engagement'])
    st.pyplot(fig)