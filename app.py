import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.title("Trend Analysis")

model = joblib.load("model.pkl")

file = st.file_uploader("Upload Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    # =========================
    # PROCESSING
    # =========================
    df['date'] = pd.to_datetime(df[['Year','Month','Day','Hour']])
    df = df.sort_values('date')
    df['engagement'] = df['Likes'] + df['Retweets']

    # =========================
    # SPLIT INTO 4 PARTS
    # =========================
    df = df.reset_index(drop=True)
    parts = np.array_split(df, 4)
    part3 = parts[2]
    part4 = parts[3]

    # =========================
    # CURRENT TREND
    # =========================
    st.subheader("Current Trends")

    current = part4.groupby('Hashtags')['engagement'].sum().sort_values(ascending=False)
    st.write(current.head(5))

    # =========================
    # TREND DIRECTION
    # =========================
    prev = part3.groupby('Hashtags')['engagement'].sum()
    latest = part4.groupby('Hashtags')['engagement'].sum()

    trend = (latest - prev).fillna(0).sort_values(ascending=False)

    st.subheader("Rising Trends")
    st.write(trend.head(3))

    st.subheader("Falling Trends")
    st.write(trend.tail(3))

    # =========================
    # FEATURES FOR MODEL
    # =========================
    df['day_number'] = (df['date'] - df['date'].min()).dt.days
    df['moving_avg'] = df['engagement'].rolling(7).mean().fillna(0)
    df['growth_rate'] = df['engagement'].pct_change().fillna(0)

    X = df[['day_number','moving_avg','growth_rate']]

    # =========================
    # FUTURE PREDICTION
    # =========================
    future_days = np.arange(df['day_number'].max()+1, df['day_number'].max()+8)

    future_df = pd.DataFrame({
        'day_number': future_days,
        'moving_avg': [df['moving_avg'].iloc[-1]]*7,
        'growth_rate': [df['growth_rate'].iloc[-1]]*7
    })

    preds = model.predict(future_df)

    st.subheader("Future Prediction")
    st.write(preds)

    # =========================
    # GRAPH
    # =========================
    st.subheader("Engagement Over Time")

    plt.figure()
    plt.plot(df['date'], df['engagement'])
    plt.xticks(rotation=45)
    st.pyplot(plt)