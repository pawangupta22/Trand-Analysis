import streamlit as st
import numpy as np
from model import load_and_train

st.set_page_config(page_title="Trend Predictor", layout="wide")

st.title("🚀 Social Media Trend Dashboard")

# Load models
df, rf_reg, rf_clf, le_platform, le_region, le_content, le_hashtag = load_and_train()

# -------------------------------
# SECTION 1: CURRENT TRENDING
# -------------------------------
st.header("🔥 Current Trending Hashtags")

recent = df[df['Post_Date'] >= df['Post_Date'].max() - pd.Timedelta(days=30)]
trending = recent.groupby('Hashtag')['Views'].sum().sort_values(ascending=False)

st.bar_chart(trending.head(10))

# -------------------------------
# SECTION 2: USER INPUT
# -------------------------------
st.header("🎯 Predict Your Content Performance")

col1, col2 = st.columns(2)

with col1:
    platform = st.selectbox("Platform", le_platform.classes_)
    content = st.selectbox("Content Type", le_content.classes_)
    region = st.selectbox("Region", le_region.classes_)

with col2:
    hashtag = st.selectbox("Hashtag", le_hashtag.classes_)
    day = st.slider("Day", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)
    dayofweek = st.slider("Day of Week (0=Mon)", 0, 6, 3)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("🔮 Predict"):

    # Encode
    platform_enc = le_platform.transform([platform])[0]
    region_enc = le_region.transform([region])[0]
    content_enc = le_content.transform([content])[0]
    hashtag_enc = le_hashtag.transform([hashtag])[0]

    user_data = np.array([[platform_enc, region_enc, content_enc, day, month, dayofweek, hashtag_enc]])

    # Predict
    views = rf_reg.predict(user_data)[0]
    trend = rf_clf.predict(user_data)[0]

    # Output
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Expected Views", int(views))

    with col2:
        if trend == "Trending":
            st.success("🚀 Trending")
        elif trend == "Falling":
            st.error("📉 Falling")
        else:
            st.warning("😐 Stable")

# -------------------------------
# SECTION 3: PATTERN INSIGHTS
# -------------------------------
st.header("📊 Feature Importance (What Matters Most)")

importance = rf_reg.feature_importances_
features = ['Platform','Region','Content_Type','Day','Month','DayOfWeek','Hashtag_enc']

import pandas as pd
imp_df = pd.DataFrame({"Feature": features, "Importance": importance})
imp_df = imp_df.sort_values(by="Importance", ascending=False)

st.bar_chart(imp_df.set_index("Feature"))