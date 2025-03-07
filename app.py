import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
import numpy as np
from sklearn.cluster import KMeans

# ---------------------------
# 1️⃣ Load Data Function (Fixed File Paths)
# ---------------------------
@st.cache_data
def load_data():
    file_paths = ["amazon.csv", "Clustered Data.csv"]
    
    # Check if files exist
    missing_files = [file for file in file_paths if not os.path.exists(file)]
    if missing_files:
        st.error(f"❌ ERROR: Missing files: {', '.join(missing_files)}. Make sure all datasets are uploaded to GitHub.")
        return None, None

    # Load datasets
    amazon_data = pd.read_csv("amazon.csv")
    clustered_data = pd.read_csv("Clustered Data.csv")
    
    return amazon_data, clustered_data

# Load data
df, clustered_df = load_data()

# Convert price columns to numeric and handle missing values
df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')
df['discounted_price'] = pd.to_numeric(df['discounted_price'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
df.fillna({'actual_price': 0, 'discounted_price': 0, 'rating_count': 0}, inplace=True)

# Ensure actual_price is not zero to avoid division by zero
df['discount_percentage'] = np.where(
    df['actual_price'] > 0, 
    ((df['actual_price'] - df['discounted_price']) / df['actual_price']) * 100, 
    0
)

# ---------------------------
# 2️⃣ Streamlit Sidebar Navigation
# ---------------------------
st.sidebar.title("📊 Amazon E-Commerce Insights")
page = st.sidebar.radio("Choose Analysis", ["Customer Segments", "User Behavior", "Product Analysis", "Exploratory Data Analysis"])

# ---------------------------
# 3️⃣ Customer Segmentation Page
# ---------------------------
if page == "Customer Segments":
    st.title("📌 Customer Segmentation")
    if clustered_df is not None:
        st.write("This section presents customer groups based on shopping behavior.")
        
        # Clustering Distribution
        st.write("### Cluster Distribution")
        fig = px.histogram(clustered_df, x='cluster', color='cluster', title="Customer Segments Distribution")
        st.plotly_chart(fig)

        # Scatter Plot of Clusters
        st.write("### Cluster Visualization")
        fig = px.scatter(clustered_df, x='actual_price', y='discounted_price', color='cluster', title="Customer Segments Clustering")
        st.plotly_chart(fig)

# ---------------------------
# 4️⃣ User Behavior Analysis Page
# ---------------------------
elif page == "User Behavior":
    st.title("📌 User Behavior Analysis")
    if df is not None:
        st.write("This section analyzes customer reviews and engagement trends.")
        
        # Customer Sentiment Distribution
        df['sentiment'] = df['review_content'].fillna(" ").apply(lambda x: "Positive" if "good" in x else "Negative" if "bad" in x else "Neutral")
        sentiment_counts = df['sentiment'].value_counts()
        
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Customer Sentiment Distribution")
        st.plotly_chart(fig)

        # Word Cloud
        st.write("### Word Cloud of Reviews")
        all_reviews = " ".join(df['review_content'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

# ---------------------------
# 5️⃣ Exploratory Data Analysis Page
# ---------------------------
elif page == "Exploratory Data Analysis":
    st.title("📌 Exploratory Data Analysis (EDA)")
    if df is not None:
        st.write("### Distribution of Discounted and Actual Prices")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        sns.histplot(df['discounted_price'], bins=30, kde=True, ax=axes[0, 0], color='blue')
        sns.boxplot(x=df['discounted_price'], ax=axes[0, 1], color='blue')
        sns.histplot(df['actual_price'], bins=30, kde=True, ax=axes[1, 0], color='red')
        sns.boxplot(x=df['actual_price'], ax=axes[1, 1], color='red')
        st.pyplot(fig)

        st.write("### Scatter Plot: Actual Price vs Discounted Price with Discount Percentage")
        fig = px.scatter(df, x='actual_price', y='discounted_price', color='discount_percentage', title="Actual Price vs Discounted Price", size_max=10)
        st.plotly_chart(fig)

        st.write("### Heatmap of Correlations")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[['discounted_price', 'actual_price', 'rating', 'rating_count', 'discount_percentage']].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

# ---------------------------
# 📌 Footer
# ---------------------------
st.sidebar.info("Built with Streamlit | AI-Powered E-Commerce Insights")
