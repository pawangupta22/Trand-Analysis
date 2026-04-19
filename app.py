# =============================================================
# SOCIAL MEDIA TREND ANALYSIS - STREAMLIT DASHBOARD
# Run: streamlit run app.py
# Make sure model.pkl is in the same folder as this file
# =============================================================

# =========================
# 1. IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# =========================
# 2. PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Social Media Trend Analysis",
    layout="wide"
)

st.title("Social Media Trend Analysis Dashboard")
st.markdown("Upload your dataset to analyze hashtag trends and predict future engagement.")

# =========================
# 3. LOAD MODEL
# =========================
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.warning(
        "model.pkl not found. Please run colab_train.py first to generate the model, "
        "then place model.pkl in the same folder as this app."
    )

# =========================
# 4. FILE UPLOAD
# =========================
file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if not file:
    st.info("Please upload a CSV file to begin analysis.")
    st.stop()

# =========================
# 5. DATA CLEANING
# =========================
@st.cache_data
def load_and_clean(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Check required columns
    required = ['Year', 'Month', 'Day', 'Hour', 'Likes', 'Retweets', 'Hashtags']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, f"Missing required columns: {missing}"

    # Strip whitespace from string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # Convert to datetime
    df['date'] = pd.to_datetime(
        df[['Year', 'Month', 'Day', 'Hour']],
        errors='coerce'
    )

    # Drop invalid dates
    original_len = len(df)
    df = df.dropna(subset=['date'])
    dropped = original_len - len(df)

    if len(df) == 0:
        return None, "No valid rows remain after date parsing."

    df = df.sort_values('date').reset_index(drop=True)
    return df, f"Loaded {len(df)} valid rows. ({dropped} rows dropped due to invalid dates.)"

df, message = load_and_clean(file)

if df is None:
    st.error(message)
    st.stop()
else:
    st.caption(message)

# =========================
# 6. DATA PREPARATION
# =========================
df['engagement'] = df['Likes'] + df['Retweets']
df['day_number'] = (df['date'] - df['date'].min()).dt.days
df['moving_avg'] = df['engagement'].rolling(7, min_periods=1).mean()
df['growth_rate'] = df['engagement'].pct_change().fillna(0).clip(-5, 5)

# =========================
# 7. SPLIT: OLD 75% / LATEST 25%
# =========================
n = len(df)
split_idx = int(n * 0.75)

if split_idx == 0 or split_idx == n:
    st.error("Dataset too small to split into old/latest portions. Please use a larger dataset.")
    st.stop()

old_data = df.iloc[:split_idx]
latest_data = df.iloc[split_idx:]

# =========================
# 8. DATASET OVERVIEW
# =========================
st.subheader("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", n)
col2.metric("Date Range Start", df['date'].min().strftime("%b %d, %Y"))
col3.metric("Date Range End", df['date'].max().strftime("%b %d, %Y"))
col4.metric("Unique Hashtags", df['Hashtags'].nunique())

st.markdown("---")

# =========================
# 9. CURRENT TRENDING HASHTAGS
# =========================
st.subheader("Current Trending Hashtags (Latest 25% of Data)")

current_trends = (
    latest_data.groupby('Hashtags')['engagement']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
current_trends.columns = ['Hashtag', 'Total Engagement']
current_trends.index = current_trends.index + 1

col_left, col_right = st.columns([1, 1])

with col_left:
    st.dataframe(current_trends, use_container_width=True)

with col_right:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(
        current_trends['Hashtag'][::-1],
        current_trends['Total Engagement'][::-1],
        color='steelblue'
    )
    ax.set_xlabel("Total Engagement")
    ax.set_title("Top 10 Current Hashtags")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# =========================
# 10. TREND DIRECTION ANALYSIS
# =========================
st.subheader("Trend Direction: Rising vs Falling")

old_eng = old_data.groupby('Hashtags')['engagement'].sum()
latest_eng = latest_data.groupby('Hashtags')['engagement'].sum()

# Align on common hashtags
all_tags = old_eng.index.union(latest_eng.index)
old_aligned = old_eng.reindex(all_tags, fill_value=0)
latest_aligned = latest_eng.reindex(all_tags, fill_value=0)

trend_diff = (latest_aligned - old_aligned).sort_values(ascending=False)

# Label direction
def label_trend(val):
    if val > 0:
        return "Rising"
    elif val < 0:
        return "Falling"
    else:
        return "Stable"

trend_df = pd.DataFrame({
    'Hashtag': trend_diff.index,
    'Engagement Change': trend_diff.values,
    'Direction': trend_diff.values
}).assign(Direction=lambda x: x['Direction'].apply(label_trend))
trend_df = trend_df.reset_index(drop=True)

col_rise, col_fall = st.columns(2)

with col_rise:
    st.markdown("**Rising Trends (Top 5)**")
    rising = trend_df[trend_df['Direction'] == 'Rising'].head(5).reset_index(drop=True)
    rising.index = rising.index + 1
    st.dataframe(rising[['Hashtag', 'Engagement Change']], use_container_width=True)

with col_fall:
    st.markdown("**Falling Trends (Top 5)**")
    falling = trend_df[trend_df['Direction'] == 'Falling'].tail(5).sort_values(
        'Engagement Change'
    ).reset_index(drop=True)
    falling.index = falling.index + 1
    st.dataframe(falling[['Hashtag', 'Engagement Change']], use_container_width=True)

st.markdown("---")

# =========================
# 11. FUTURE ENGAGEMENT PREDICTION
# =========================
st.subheader("Future Engagement Prediction (Next 7 Time Steps)")

if model is None:
    st.warning("No model loaded. Showing linear trend estimate instead.")

    # Simple linear fallback
    last_val = df['engagement'].iloc[-1]
    avg_delta = df['engagement'].diff().mean()
    preds = [last_val + avg_delta * i for i in range(1, 8)]
else:
    last_day = df['day_number'].max()
    last_moving_avg = df['moving_avg'].iloc[-1]
    last_growth_rate = df['growth_rate'].iloc[-1]

    future_df = pd.DataFrame({
        'day_number': np.arange(last_day + 1, last_day + 8),
        'moving_avg': [last_moving_avg] * 7,
        'growth_rate': [last_growth_rate] * 7
    })
    preds = model.predict(future_df)

pred_df = pd.DataFrame({
    'Step': [f"Step {i+1}" for i in range(7)],
    'Predicted Engagement': np.round(preds, 1)
})
pred_df.index = pred_df.index + 1

col_pred_table, col_pred_chart = st.columns([1, 1])

with col_pred_table:
    st.dataframe(pred_df, use_container_width=True)

with col_pred_chart:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(
        pred_df['Step'],
        pred_df['Predicted Engagement'],
        marker='o',
        color='darkorange',
        linewidth=2
    )
    ax2.set_xlabel("Future Step")
    ax2.set_ylabel("Predicted Engagement")
    ax2.set_title("Predicted Engagement - Next 7 Steps")
    ax2.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.markdown("---")

# =========================
# 12. ENGAGEMENT OVER TIME GRAPH
# =========================
st.subheader("Engagement Over Time")

# Resample to daily for cleaner chart
daily_eng = df.set_index('date')['engagement'].resample('D').sum().reset_index()
daily_eng.columns = ['date', 'engagement']

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(daily_eng['date'], daily_eng['engagement'], color='steelblue', linewidth=1.2)
ax3.fill_between(daily_eng['date'], daily_eng['engagement'], alpha=0.15, color='steelblue')

# Vertical split marker
split_date = old_data['date'].max()
ax3.axvline(x=split_date, color='red', linestyle='--', linewidth=1, label='Old / Latest split')

ax3.set_xlabel("Date")
ax3.set_ylabel("Total Engagement")
ax3.set_title("Daily Engagement Over Time")
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax3.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
ax3.legend()
plt.tight_layout()
st.pyplot(fig3)
plt.close()

# Moving average overlay
st.markdown("**7-Day Rolling Average of Engagement**")
fig4, ax4 = plt.subplots(figsize=(12, 3))
ax4.plot(df['date'], df['moving_avg'], color='seagreen', linewidth=1.5)
ax4.set_xlabel("Date")
ax4.set_ylabel("Moving Avg Engagement")
ax4.set_title("7-Day Rolling Mean")
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax4.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig4)
plt.close()

st.markdown("---")

# =========================
# 13. RAW DATA PREVIEW
# =========================
with st.expander("View Raw Data (first 50 rows)"):
    st.dataframe(
        df[['date', 'Hashtags', 'Likes', 'Retweets', 'engagement']].head(50),
        use_container_width=True
    )