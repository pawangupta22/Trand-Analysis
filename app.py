# ============================================================
# SOCIAL MEDIA TREND ANALYSIS - STREAMLIT DASHBOARD
# Requires: model.pkl (trained in Colab) in same folder
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ---- PAGE SETUP ----
st.set_page_config(page_title="Trend Analysis", layout="wide")
st.title("Social Media Trend Analysis")
st.caption("Instagram India 2025 | Hashtag Engagement Tracker")

# ---- LOAD MODEL ----
if not os.path.exists("model.pkl"):
    st.error("model.pkl not found. Please train the model in Colab first and place model.pkl in this folder.")
    st.stop()

model = joblib.load("model.pkl")

# ---- FILE UPLOAD ----
uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])
if not uploaded:
    st.info("Please upload the dataset CSV to begin.")
    st.stop()

# ============================================================
# LOAD & CLEAN
# ============================================================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes('object').columns:
        df[col] = df[col].str.strip()
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    df['engagement'] = df['Likes'] + df['Retweets']
    df['day_number']  = (df['date'] - df['date'].min()).dt.days
    df['moving_avg']  = df['engagement'].rolling(7, min_periods=1).mean()
    df['growth_rate'] = df['engagement'].pct_change().fillna(0).clip(-5, 5)
    return df

df = load_data(uploaded)

if len(df) < 4:
    st.error("Dataset too small. Need at least 4 rows.")
    st.stop()

# ============================================================
# SPLIT INTO 4 PARTS
# ============================================================
parts = np.array_split(df, 4)
part1, part2, part3, part4 = parts
old_data    = pd.concat([part1, part2, part3])
latest_data = part4

# ============================================================
# SECTION 1 — DATASET OVERVIEW
# ============================================================
st.header("1. Dataset Overview")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Records",   f"{len(df):,}")
c2.metric("Unique Hashtags", df['Hashtags'].nunique())
c3.metric("Date From",       df['date'].min().strftime("%d %b %Y"))
c4.metric("Date To",         df['date'].max().strftime("%d %b %Y"))
c5.metric("Avg Engagement",  f"{df['engagement'].mean():.0f}")

st.markdown("---")

# ============================================================
# SECTION 2 — 4-PART SPLIT SUMMARY
# ============================================================
st.header("2. Dataset Split into 4 Parts")

split_rows = []
for i, p in enumerate([part1, part2, part3, part4], 1):
    split_rows.append({
        "Part":           f"Part {i}" + (" (Latest)" if i == 4 else " (Old)"),
        "Rows":           f"{len(p):,}",
        "From":           p['date'].min().strftime("%d %b %Y"),
        "To":             p['date'].max().strftime("%d %b %Y"),
        "Avg Engagement": f"{p['engagement'].mean():.0f}",
        "Top Hashtag":    p.groupby('Hashtags')['engagement'].sum().idxmax()
    })

st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)
st.markdown("---")

# ============================================================
# SECTION 3 — TREND DIRECTION
# ============================================================
st.header("3. Trend Direction: Latest vs Old Parts")

old_eng    = old_data.groupby('Hashtags')['engagement'].mean()
latest_eng = latest_data.groupby('Hashtags')['engagement'].mean()
all_tags   = old_eng.index.union(latest_eng.index)
old_al     = old_eng.reindex(all_tags, fill_value=0)
lat_al     = latest_eng.reindex(all_tags, fill_value=0)
pct        = ((lat_al - old_al) / old_al.replace(0, np.nan) * 100).fillna(0)

def label(v):
    if v > 3:    return "Rising"
    elif v < -3: return "Falling"
    else:        return "Stable"

trend_df = pd.DataFrame({
    "Hashtag":        all_tags,
    "Old Avg Eng":    old_al.values.round(0).astype(int),
    "Latest Avg Eng": lat_al.values.round(0).astype(int),
    "Change %":       pct.values.round(1),
    "Trend":          pct.values
}).assign(Trend=lambda x: x['Trend'].apply(label)).sort_values("Change %", ascending=False)

rising  = trend_df[trend_df['Trend'] == "Rising"]
stable  = trend_df[trend_df['Trend'] == "Stable"]
falling = trend_df[trend_df['Trend'] == "Falling"].sort_values("Change %")

tab1, tab2, tab3 = st.tabs([
    f"Rising  ({len(rising)})",
    f"Stable  ({len(stable)})",
    f"Falling ({len(falling)})"
])
with tab1:
    st.dataframe(rising[['Hashtag','Old Avg Eng','Latest Avg Eng','Change %','Trend']],
                 use_container_width=True, hide_index=True)
with tab2:
    st.dataframe(stable[['Hashtag','Old Avg Eng','Latest Avg Eng','Change %','Trend']],
                 use_container_width=True, hide_index=True)
with tab3:
    st.dataframe(falling[['Hashtag','Old Avg Eng','Latest Avg Eng','Change %','Trend']],
                 use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================================
# SECTION 4 — PREDICTION
# ============================================================
st.header("4. Engagement Prediction (Next 7 Days)")

last_day     = int(df['day_number'].max())
last_mov_avg = float(df['moving_avg'].iloc[-1])
last_growth  = float(df['growth_rate'].iloc[-1])

future_input = pd.DataFrame({
    'day_number':  np.arange(last_day + 1, last_day + 8),
    'moving_avg':  [last_mov_avg] * 7,
    'growth_rate': [last_growth]  * 7
})
preds = model.predict(future_input)

pred_df = pd.DataFrame({
    "Day":                  [f"Day +{i}" for i in range(1, 8)],
    "Predicted Engagement": np.round(preds, 0).astype(int)
})

col_a, col_b = st.columns([1, 2])
with col_a:
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
with col_b:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(pred_df['Day'], pred_df['Predicted Engagement'],
            marker='o', linewidth=2, color='#1f77b4')
    ax.fill_between(pred_df['Day'], pred_df['Predicted Engagement'],
                    alpha=0.1, color='#1f77b4')
    ax.set_ylabel("Predicted Engagement")
    ax.set_title("Next 7 Days Forecast")
    ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ============================================================
# SECTION 5 — ENGAGEMENT OVER TIME
# ============================================================
st.header("5. Engagement Over Time")

daily = df.set_index('date')['engagement'].resample('D').sum().reset_index()

fig2, ax2 = plt.subplots(figsize=(12, 3.5))
ax2.plot(daily['date'], daily['engagement'], linewidth=1.2, color='steelblue')
ax2.fill_between(daily['date'], daily['engagement'], alpha=0.12, color='steelblue')
ax2.axvline(latest_data['date'].min(), color='red', linestyle='--',
            linewidth=1.2, label='Latest 25% starts here')
ax2.set_xlabel("Date")
ax2.set_ylabel("Daily Engagement")
ax2.set_title("Daily Total Engagement")
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.subheader("Monthly Average Engagement — Top 5 Hashtags")

top5   = df.groupby('Hashtags')['engagement'].sum().nlargest(5).index.tolist()
df_top = df[df['Hashtags'].isin(top5)].copy()
df_top['month'] = df_top['date'].dt.to_period('M').astype(str)
monthly = df_top.groupby(['month','Hashtags'])['engagement'].mean().unstack(fill_value=0)

fig3, ax3 = plt.subplots(figsize=(12, 4))
for col in monthly.columns:
    ax3.plot(monthly.index, monthly[col], marker='o', markersize=3,
             label=col, linewidth=1.5)
ax3.set_xlabel("Month")
ax3.set_ylabel("Avg Engagement")
ax3.set_title("Monthly Engagement Trend — Top 5 Hashtags")
ax3.legend(loc='upper left', fontsize=8)
ax3.tick_params(axis='x', rotation=35)
plt.tight_layout()
st.pyplot(fig3)
plt.close()