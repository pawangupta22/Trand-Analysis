import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_and_train():
    df = pd.read_csv("Cleaned_Viral_Social_Media_Trends.csv")

    df['Post_Date'] = pd.to_datetime(df['Post_Date'])
    df = df.sort_values(by='Post_Date')

    # Feature engineering
    df['Day'] = df['Post_Date'].dt.day
    df['Month'] = df['Post_Date'].dt.month
    df['DayOfWeek'] = df['Post_Date'].dt.dayofweek

    # Encoders
    le_platform = LabelEncoder()
    le_region = LabelEncoder()
    le_content = LabelEncoder()
    le_hashtag = LabelEncoder()

    df['Platform'] = le_platform.fit_transform(df['Platform'])
    df['Region'] = le_region.fit_transform(df['Region'])
    df['Content_Type'] = le_content.fit_transform(df['Content_Type'])
    df['Hashtag_enc'] = le_hashtag.fit_transform(df['Hashtag'])

    # Trend label
    df['Trend_Label'] = np.where(df['Views'].diff() > 0, 'Trending',
                         np.where(df['Views'].diff() < 0, 'Falling', 'Stable'))
    df = df.dropna()

    features = ['Platform','Region','Content_Type','Day','Month','DayOfWeek','Hashtag_enc']

    X = df[features]
    y_views = df['Views']
    y_trend = df['Trend_Label']

    # Train models
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_reg.fit(X, y_views)
    rf_clf.fit(X, y_trend)

    return df, rf_reg, rf_clf, le_platform, le_region, le_content, le_hashtag