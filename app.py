import streamlit as st
import numpy as np
import pandas as pd
from model import load_and_train

st.set_page_config(page_title="Trend Dashboard", layout="wide")

# -------------------------------
# CLEAN DARK UI FIX
# -------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

section[data-testid="stSidebar"] {
    background-color: #111;
}

h1, h2, h3 {
    margin-bottom: 0.5rem;
}

/* Remove white strip effect */
.stMetric {
    background: #1c1c1c;
    padding: 15px;
    border-radius: 10px;
}

div[data-testid="stDataFrame"] {
    background: #1c1c1c;
    border-radius: 10px;
}

/* Fix spacing */
.stPlotlyChart, .stBarChart {
    padding: 0 !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Social Media Trend Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
df, rf_reg, rf_clf, le_platform, le_region, le_content, le_hashtag = load_and_train()

# Data Range
start_year = df['Post_Date'].min().year
end_year = df['Post_Date'].max().year

st.info(f"Data Available: {start_year} to {end_year}")

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("Input")

platform = st.sidebar.selectbox("Platform", le_platform.classes_)
content = st.sidebar.selectbox("Content Type", le_content.classes_)
region = st.sidebar.selectbox("Region", le_region.classes_)
hashtag = st.sidebar.selectbox("Hashtag", le_hashtag.classes_)

day = st.sidebar.slider("Day", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 6)
dayofweek = st.sidebar.slider("Day of Week", 0, 6, 3)

predict_btn = st.sidebar.button("Predict")

# -------------------------------
# SECTION 1: TRENDING
# -------------------------------
st.subheader("Current Trending")

col1, col2 = st.columns([3, 1])

recent = df[df['Post_Date'] >= df['Post_Date'].max() - pd.Timedelta(days=30)]
trending = recent.groupby('Hashtag')['Views'].sum().sort_values(ascending=False)

with col1:
    st.bar_chart(trending.head(10))

with col2:
    st.dataframe(trending.head(10), use_container_width=True)

# -------------------------------
# SECTION 2: PREDICTION
# -------------------------------
st.subheader("Prediction")

if predict_btn:

    platform_enc = le_platform.transform([platform])[0]
    region_enc = le_region.transform([region])[0]
    content_enc = le_content.transform([content])[0]
    hashtag_enc = le_hashtag.transform([hashtag])[0]

    user_data = np.array([[platform_enc, region_enc, content_enc, day, month, dayofweek, hashtag_enc]])

    views = rf_reg.predict(user_data)[0]
    trend = rf_clf.predict(user_data)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Expected Views", int(views))

    with col2:
        st.metric("Trend", trend)

# -------------------------------
# SECTION 3: FEATURE IMPORTANCE
# -------------------------------
st.subheader("Feature Importance")

importance = rf_reg.feature_importances_
features = ['Platform','Region','Content_Type','Day','Month','DayOfWeek','Hashtag_enc']

imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(imp_df.set_index("Feature"))
