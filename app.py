import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(page_title="AI Data Analytics & Prediction App", layout="wide")

st.title("🚢 Titanic AI Analytics & Survival Predictor")
st.markdown("An interactive End-to-End Data Science & Machine Learning Application")

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('sales_data.csv')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("🎯 Filter Data")
selected_class = st.sidebar.multiselect("Passenger Class", options=[1, 2, 3], default=[1, 2, 3])
selected_gender = st.sidebar.multiselect("Gender", options=df['Sex'].unique(), default=df['Sex'].unique())

filtered_df = df[(df['Pclass'].isin(selected_class)) & (df['Sex'].isin(selected_gender))]

# Top Key Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Passengers", len(filtered_df))
col2.metric("Survival Rate", f"{filtered_df['Survived'].mean() * 100:.1f}%")
col3.metric("Avg Fare Paid", f"${filtered_df['Fare'].mean():.2f}")
col4.metric("Avg Age", f"{filtered_df['Age'].mean():.1f} yrs")

st.divider()

# Visual Analytics Section
st.subheader("📊 Interactive Business Visualizations")
tab1, tab2 = st.tabs(["Class vs Survival Analysis", "Fare vs Age Distribution"])

with tab1:
    fig_bar = px.histogram(filtered_df, x="Pclass", color="Survived", barmode="group",
                           title="Survival Count by Ticket Class",
                           color_discrete_map={0: '#EF553B', 1: '#00CC96'})
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    fig_scatter = px.scatter(filtered_df, x="Age", y="Fare", color="Survived",
                             size="Fare", hover_data=['Name'],
                             title="Age vs Ticket Fare Correlation")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# Machine Learning Section
st.subheader("🤖 Live AI Survival Prediction Engine")
st.markdown("Test the Machine Learning Model in real-time by entering passenger details:")

# Model Prep
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
y = df['Survived']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# User Inputs for Prediction
p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    in_pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3], index=0)
    in_sex = st.selectbox("Gender", ["male", "female"])
with p_col2:
    in_age = st.slider("Age", 1, 80, 28)
    in_fare = st.slider("Fare Paid ($)", 5, 500, 50)
with p_col3:
    in_sibsp = st.number_input("Siblings / Spouses Aboard", 0, 8, 0)
    in_parch = st.number_input("Parents / Children Aboard", 0, 6, 0)

if st.button("🚀 Run AI Prediction"):
    sex_num = 1 if in_sex == "female" else 0
    input_data = np.array([[in_pclass, sex_num, in_age, in_sibsp, in_parch, in_fare]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.success(f"🎉 **High Chance of Survival!** (Confidence: {prob:.1f}%)")
    else:
        st.error(f"⚠️ **Low Chance of Survival.** (Confidence: {100 - prob:.1f}%)")
