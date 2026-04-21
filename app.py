import streamlit as st
import numpy as np
import pandas as pd
from model import load_and_train

st.set_page_config(page_title="Trend Dashboard", layout="wide")

# -------------------------------
# CUSTOM STYLE (CLEAN UI)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #222;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

st.title("Social Media Trend Analysis Dashboard")

# -------------------------------
# LOAD MODEL
# -------------------------------
df, rf_reg, rf_clf, le_platform, le_region, le_content, le_hashtag = load_and_train()

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("Input Parameters")

platform = st.sidebar.selectbox("Platform", le_platform.classes_)
content = st.sidebar.selectbox("Content Type", le_content.classes_)
region = st.sidebar.selectbox("Region", le_region.classes_)
hashtag = st.sidebar.selectbox("Hashtag", le_hashtag.classes_)

day = st.sidebar.slider("Day", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 6)
dayofweek = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 3)

predict_btn = st.sidebar.button("Predict")

# -------------------------------
# SECTION 1: TRENDING
# -------------------------------
st.subheader("Current Trending Hashtags")

col1, col2 = st.columns([2, 1])

with col1:
    recent = df[df['Post_Date'] >= df['Post_Date'].max() - pd.Timedelta(days=30)]
    trending = recent.groupby('Hashtag')['Views'].sum().sort_values(ascending=False)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.bar_chart(trending.head(10))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Top Hashtags")
    st.dataframe(trending.head(10))
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# SECTION 2: PREDICTION OUTPUT
# -------------------------------
st.subheader("Prediction Result")

if predict_btn:

    # Encode
    platform_enc = le_platform.transform([platform])[0]
    region_enc = le_region.transform([region])[0]
    content_enc = le_content.transform([content])[0]
    hashtag_enc = le_hashtag.transform([hashtag])[0]

    user_data = np.array([[platform_enc, region_enc, content_enc, day, month, dayofweek, hashtag_enc]])

    # Predict
    views = rf_reg.predict(user_data)[0]
    trend = rf_clf.predict(user_data)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Expected Views", int(views))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Trend Status")

        if trend == "Trending":
            st.write("Growing")
        elif trend == "Falling":
            st.write("Falling")
        else:
            st.write("Stable")

        st.markdown('</div>', unsafe_allow_html=True)

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

st.markdown('<div class="card">', unsafe_allow_html=True)
st.bar_chart(imp_df.set_index("Feature"))
st.markdown('</div>', unsafe_allow_html=True)
