import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load datasets (Modify paths as needed)
@st.cache_data
def load_data():
    amazon_data = pd.read_csv("C:/Users/prava/OneDrive/Desktop/IBCP/AI IBCP/Data Minig SA/amazon.csv")
    clustered_data = pd.read_csv("C:/Users/prava/OneDrive/Desktop/IBCP/AI IBCP/Data Minig SA/Clustered Data.csv")
    association_rules = pd.read_csv("C:/Users/prava/OneDrive/Desktop/IBCP/AI IBCP/Data Minig SA/Amazon Association rules.csv")
    return amazon_data, clustered_data, association_rules

df, clustered_df, rules_df = load_data()

# Sidebar Navigation
st.sidebar.title("📊 Amazon E-Commerce Insights")
page = st.sidebar.radio("Choose Analysis", ["Customer Segments", "Frequent Itemsets", "User Behavior"])

# ---------------------------
#  Customer Segmentation Page
# ---------------------------
if page == "Customer Segments":
    st.title(" Customer Segmentation")
    st.write("This section presents customer groups based on shopping behavior.")

    # Display cluster statistics
    st.write("### Cluster Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=clustered_df['cluster'], palette="coolwarm", ax=ax)
    plt.xlabel("Customer Segments")
    plt.ylabel("Count")
    st.pyplot(fig)

    # Display sample data
    st.write("### Sample Segmented Data")
    st.dataframe(clustered_df.head())

# ---------------------------
#  Frequent Itemsets Page
# ---------------------------
elif page == "Frequent Itemsets":
    st.title(" Association Rule Mining")
    st.write("This section showcases frequent product combinations using Apriori.")

    # Display association rules
    st.write("### Top 10 Association Rules")
    st.dataframe(rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

    # Scatter plot for Lift vs Confidence
    st.write("### Confidence vs Lift of Rules")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=rules_df['confidence'], y=rules_df['lift'], alpha=0.7, ax=ax)
    plt.xlabel("Confidence")
    plt.ylabel("Lift")
    plt.title("Association Rules - Confidence vs Lift")
    st.pyplot(fig)

# ---------------------------
#  User Behavior Page
# ---------------------------
elif page == "User Behavior":
    st.title(" User Behavior Analysis")
    st.write("This section analyzes customer reviews and engagement trends.")

    # Sentiment Distribution
    st.write("### Customer Sentiment Distribution")
    df['sentiment'] = df['review_content'].fillna("").apply(lambda x: "Positive" if "good" in x else "Negative" if "bad" in x else "Neutral")
    sentiment_counts = df['sentiment'].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='coolwarm', ax=ax)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Analysis of Customer Reviews")
    st.pyplot(fig)

    # Top Reviewed Products
    st.write("### Most Reviewed Products")
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    top_products = df.groupby('product_name')['rating_count'].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_products.index, y=top_products.values, palette='magma', ax=ax)
    plt.xticks(rotation=90)
    plt.xlabel("Product Name")
    plt.ylabel("Review Count")
    st.pyplot(fig)

    # Word Cloud for Reviews
    st.write("### Word Cloud of Reviews")
    all_reviews = " ".join(df['review_content'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# ---------------------------
#  Footer Section
# ---------------------------
st.sidebar.info("Built with Streamlit | AI-Powered E-Commerce Insights")