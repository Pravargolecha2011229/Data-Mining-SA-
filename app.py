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
    file_paths = ["amazon.csv", "Clustered Data.csv", "Amazon Association rules.csv"]
    
    # Check if files exist
    missing_files = [file for file in file_paths if not os.path.exists(file)]
    if missing_files:
        st.error(f"❌ ERROR: Missing files: {', '.join(missing_files)}. Make sure all datasets are uploaded to GitHub.")
        return None, None, None

    # Load datasets
    amazon_data = pd.read_csv("amazon.csv")
    clustered_data = pd.read_csv("Clustered Data.csv")
    association_rules = pd.read_csv("Amazon Association rules.csv")
    
    return amazon_data, clustered_data, association_rules

# Load data
df, clustered_df, rules_df = load_data()

# Convert price columns to numeric and handle missing values
df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')
df['discounted_price'] = pd.to_numeric(df['discounted_price'], errors='coerce')
df.fillna({'actual_price': 0, 'discounted_price': 0}, inplace=True)

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
page = st.sidebar.radio("Choose Analysis", ["Customer Segments", "Frequent Itemsets", "User Behavior", "Product Analysis", "Exploratory Data Analysis"])

# ---------------------------
# 3️⃣ Customer Segmentation Page
# ---------------------------
if page == "Customer Segments":
    st.title("📌 Customer Segmentation")
    if clustered_df is not None:
        st.write("This section presents customer groups based on shopping behavior.")
        
        # Elbow Method for Optimal K
        distortions = []
        K = range(1, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(clustered_df[['discounted_price', 'actual_price']])
            distortions.append(kmeans.inertia_)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title('Elbow Method for Optimal k')
        st.pyplot(fig)
        
        # Clustering Distribution
        st.write("### Cluster Distribution")
        fig = px.histogram(clustered_df, x='cluster', color='cluster', title="Customer Segments Distribution")
        st.plotly_chart(fig)

        # Scatter Plot of Clusters
        st.write("### Cluster Visualization")
        fig = px.scatter(clustered_df, x='actual_price', y='discounted_price', color='cluster', title="Customer Segments Clustering")
        st.plotly_chart(fig)

# ---------------------------
# 4️⃣ Frequent Itemsets Page
# ---------------------------
elif page == "Frequent Itemsets":
    st.title("📌 Association Rule Mining")
    if rules_df is not None:
        st.write("This section showcases frequent product combinations using Apriori.")

        # Display top association rules
        st.write("### Top 10 Association Rules")
        st.dataframe(rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

        # Scatter plot for Lift vs Confidence
        st.write("### Confidence vs Lift of Rules")
        fig = px.scatter(rules_df, x='confidence', y='lift', title="Association Rules - Confidence vs Lift", size='support', color='lift')
        st.plotly_chart(fig)

# ---------------------------
# 5️⃣ User Behavior Analysis Page
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

        # Top Reviewers
        st.write("### Top 10 Reviewers")
        top_reviewers = df['user_name'].value_counts().head(10)
        fig = px.bar(x=top_reviewers.index, y=top_reviewers.values, labels={'x': 'User Name', 'y': 'Number of Reviews'}, title="Top 10 Reviewers")
        st.plotly_chart(fig)

 # ---------------------------
# 3️⃣ Exploratory Data Analysis Page
# ---------------------------
if page == "Exploratory Data Analysis":
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

        st.write("### Distribution of Product Ratings")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x=df['rating'], palette='viridis', ax=ax)
        st.pyplot(fig)

        st.write("### Top 10 Most Rated Products")
        top_products = df.nlargest(10, 'rating_count')
        fig = px.bar(top_products, x='product_name', y='rating_count', title="Top 10 Most Rated Products", labels={'rating_count': 'Number of Ratings'})
        st.plotly_chart(fig)

        st.write("### Distribution of Product Categories")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y=df['category'], order=df['category'].value_counts().index, palette='coolwarm', ax=ax)
        st.pyplot(fig)

        st.write("### Category Distribution in Pie Chart")
        fig = px.pie(df, names='category', title="Category Distribution")
        st.plotly_chart(fig)

        st.write("### Heatmap of Correlations")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[['discounted_price', 'actual_price', 'rating', 'rating_count', 'discount_percentage']].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

# ---------------------------
# 📌 Footer
# ---------------------------
st.sidebar.info("Built with Streamlit | AI-Powered E-Commerce Insights")
