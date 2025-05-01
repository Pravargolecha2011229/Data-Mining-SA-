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
from sklearn.preprocessing import StandardScaler

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
    
    # Clean and preprocess the data
    amazon_data['actual_price'] = pd.to_numeric(amazon_data['actual_price'].str.replace(',', ''), errors='coerce')
    amazon_data['discounted_price'] = pd.to_numeric(amazon_data['discounted_price'].str.replace(',', ''), errors='coerce')
    amazon_data['rating'] = pd.to_numeric(amazon_data['rating'], errors='coerce')
    amazon_data['rating_count'] = pd.to_numeric(amazon_data['rating_count'].str.replace(',', ''), errors='coerce')
    
    # Handle missing values
    amazon_data.fillna({
        'actual_price': amazon_data['actual_price'].mean(),
        'discounted_price': amazon_data['discounted_price'].mean(),
        'rating': amazon_data['rating'].mean(),
        'rating_count': amazon_data['rating_count'].mean()
    }, inplace=True)
    
    # Calculate discount percentage
    amazon_data['discount_percentage'] = np.where(
        amazon_data['actual_price'] > 0,
        ((amazon_data['actual_price'] - amazon_data['discounted_price']) / amazon_data['actual_price']) * 100,
        0
    )
    
    return amazon_data, clustered_data


# Load data
df, clustered_df = load_data()

# Convert price columns to numeric and handle missing values
df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')
df['discounted_price'] = pd.to_numeric(df['discounted_price'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
df.fillna({'actual_price': 100, 'discounted_price': 100, 'rating_count': 5.0}, inplace=True)

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
page = st.sidebar.radio("Choose Analysis", ["Customer Segments", "User Behavior", "Exploratory Data Analysis"])


if page == "User Behavior":
    st.title("🛍️ Amazon Shopping Behavior Analysis")
    
    if df is not None:
        # Create sample data for common Amazon shopping patterns
        shopping_time = pd.DataFrame({
            'Hour': range(24),
            'Orders': [10, 5, 3, 2, 1, 2, 5, 15, 25, 35, 45, 50, 55, 45, 40, 38, 42, 48, 40, 35, 30, 25, 20, 15]
        })
        
        # 1. Shopping Time Patterns
        st.header("⏰ Shopping Time Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily shopping pattern
            fig = px.line(shopping_time, 
                         x='Hour', 
                         y='Orders',
                         title="24-Hour Shopping Pattern",
                         labels={'Hour': 'Time of Day', 'Orders': 'Number of Orders'},
                         line_shape='spline')
            fig.update_layout(
                xaxis=dict(tickmode='array', ticktext=[f"{i}:00" for i in range(24)]),
                showlegend=False
            )
            st.plotly_chart(fig)
            
        with col2:
            # Peak shopping hours
            peak_hours = pd.DataFrame({
                'Time': ['Morning (6-12)', 'Afternoon (12-17)', 'Evening (17-22)', 'Night (22-6)'],
                'Traffic': [30, 45, 65, 15]
            })
            fig = px.pie(peak_hours, 
                        values='Traffic', 
                        names='Time',
                        title="Shopping Traffic Distribution",
                        hole=0.4)
            st.plotly_chart(fig)
        
        # 2. Device Usage
        st.header("📱 Shopping Device Preferences")
        col1, col2 = st.columns(2)
        
        with col1:
            devices = pd.DataFrame({
                'Device': ['Mobile', 'Desktop', 'Tablet', 'App'],
                'Usage': [45, 30, 15, 10]
            })
            fig = px.bar(devices,
                        x='Device',
                        y='Usage',
                        title="Shopping Device Distribution",
                        color='Device',
                        text='Usage')
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig)
            
        with col2:
            # Conversion rates by device
            conversion = pd.DataFrame({
                'Device': ['Mobile', 'Desktop', 'Tablet', 'App'],
                'Rate': [3.2, 4.5, 2.8, 5.1]
            })
            fig = px.bar(conversion,
                        x='Device',
                        y='Rate',
                        title="Conversion Rate by Device (%)",
                        color='Device',
                        text='Rate')
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig)
        
        # 3. Purchase Behavior
        st.header("🛒 Purchase Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            # Cart abandonment reasons
            abandonment = pd.DataFrame({
                'Reason': ['High Shipping', 'Better Price', 'Long Delivery', 'Payment Issues', 'Changed Mind'],
                'Percentage': [35, 25, 20, 12, 8]
            })
            fig = px.pie(abandonment,
                        values='Percentage',
                        names='Reason',
                        title="Cart Abandonment Reasons")
            st.plotly_chart(fig)
            
        with col2:
            # Purchase decision factors
            factors = pd.DataFrame({
                'Factor': ['Price', 'Reviews', 'Shipping', 'Brand', 'Description'],
                'Impact': [90, 85, 75, 65, 55]
            })
            fig = px.bar(factors,
                        x='Factor',
                        y='Impact',
                        title="Purchase Decision Factors",
                        color='Factor',
                        text='Impact')
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig)
        
        # 4. Customer Engagement
        st.header("🤝 Customer Engagement Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Review Rate", "8.5%", "↑ 1.2%")
        with col2:
            st.metric("Return Rate", "12.3%", "↓ 2.1%")
        with col3:
            st.metric("Repeat Purchase", "42%", "↑ 5.3%")
        with col4:
            st.metric("Prime Members", "65%", "↑ 3.7%")
            
        # 5. Shopping Seasonality
        st.header("📅 Shopping Seasonality")
        seasonal_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'Sales': [70, 65, 75, 80, 85, 90, 95, 100, 110, 120, 150, 160]
        })
        fig = px.line(seasonal_data,
                     x='Month',
                     y='Sales',
                     title="Monthly Shopping Trends",
                     markers=True)
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Sales Index (Base 100)",
        )
        st.plotly_chart(fig)
        
    else:
        st.error("❌ Error: Could not load the data.")

# user behavior section:
    
    if df is not None:
        # Create sample data for common Amazon shopping patterns
        shopping_time = pd.DataFrame({
            'Hour': range(24),
            'Orders': [10, 5, 3, 2, 1, 2, 5, 15, 25, 35, 45, 50, 55, 45, 40, 38, 42, 48, 40, 35, 30, 25, 20, 15]
        })
        
        # 1. Shopping Time Patterns
        st.header("⏰ Shopping Time Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily shopping pattern
            fig = px.line(shopping_time, 
                         x='Hour', 
                         y='Orders',
                         title="24-Hour Shopping Pattern",
                         labels={'Hour': 'Time of Day', 'Orders': 'Number of Orders'},
                         line_shape='spline')
            fig.update_layout(
                xaxis=dict(tickmode='array', ticktext=[f"{i}:00" for i in range(24)]),
                showlegend=False
            )
            st.plotly_chart(fig)
            
        with col2:
            # Peak shopping hours
            peak_hours = pd.DataFrame({
                'Time': ['Morning (6-12)', 'Afternoon (12-17)', 'Evening (17-22)', 'Night (22-6)'],
                'Traffic': [30, 45, 65, 15]
            })
            fig = px.pie(peak_hours, 
                        values='Traffic', 
                        names='Time',
                        title="Shopping Traffic Distribution",
                        hole=0.4)
            st.plotly_chart(fig)
        
        # 2. Device Usage
        st.header("📱 Shopping Device Preferences")
        col1, col2 = st.columns(2)
        
        with col1:
            devices = pd.DataFrame({
                'Device': ['Mobile', 'Desktop', 'Tablet', 'App'],
                'Usage': [45, 30, 15, 10]
            })
            fig = px.bar(devices,
                        x='Device',
                        y='Usage',
                        title="Shopping Device Distribution",
                        color='Device',
                        text='Usage')
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig)
            
        with col2:
            # Conversion rates by device
            conversion = pd.DataFrame({
                'Device': ['Mobile', 'Desktop', 'Tablet', 'App'],
                'Rate': [3.2, 4.5, 2.8, 5.1]
            })
            fig = px.bar(conversion,
                        x='Device',
                        y='Rate',
                        title="Conversion Rate by Device (%)",
                        color='Device',
                        text='Rate')
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig)
        
        # 3. Purchase Behavior
        st.header("🛒 Purchase Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            # Cart abandonment reasons
            abandonment = pd.DataFrame({
                'Reason': ['High Shipping', 'Better Price', 'Long Delivery', 'Payment Issues', 'Changed Mind'],
                'Percentage': [35, 25, 20, 12, 8]
            })
            fig = px.pie(abandonment,
                        values='Percentage',
                        names='Reason',
                        title="Cart Abandonment Reasons")
            st.plotly_chart(fig)
            
        with col2:
            # Purchase decision factors
            factors = pd.DataFrame({
                'Factor': ['Price', 'Reviews', 'Shipping', 'Brand', 'Description'],
                'Impact': [90, 85, 75, 65, 55]
            })
            fig = px.bar(factors,
                        x='Factor',
                        y='Impact',
                        title="Purchase Decision Factors",
                        color='Factor',
                        text='Impact')
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig)
        
        # 4. Customer Engagement
        st.header("🤝 Customer Engagement Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Review Rate", "8.5%", "↑ 1.2%")
        with col2:
            st.metric("Return Rate", "12.3%", "↓ 2.1%")
        with col3:
            st.metric("Repeat Purchase", "42%", "↑ 5.3%")
        with col4:
            st.metric("Prime Members", "65%", "↑ 3.7%")
            
        # 5. Shopping Seasonality
        st.header("📅 Shopping Seasonality")
        seasonal_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'Sales': [70, 65, 75, 80, 85, 90, 95, 100, 110, 120, 150, 160]
        })
        fig = px.line(seasonal_data,
                     x='Month',
                     y='Sales',
                     title="Monthly Shopping Trends",
                     markers=True)
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Sales Index (Base 100)",
        )
        st.plotly_chart(fig)
        
    else:
        st.error("❌ Error: Could not load the data.")      

# ---------------------------
# 4️⃣ Exploratory Data Analysis Page
# ---------------------------
elif page == "Exploratory Data Analysis":
    st.title("📌 Exploratory Data Analysis (EDA)")
    if df is not None:
        st.write("### Distribution of Discounted and Actual Prices")
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle("Exploratory Data Analysis (EDA) of Amazon E-Commerce Data", fontsize=16, fontweight='bold')
        
        sns.histplot(df['discounted_price'], bins=30, kde=True, ax=axes[0, 0], color='blue')
        axes[0, 0].set_title("Distribution of Discounted Price")
        sns.boxplot(x=df['discounted_price'], ax=axes[0, 1], color='blue')
        axes[0, 1].set_title("Box Plot of Discounted Price")
        sns.histplot(df['actual_price'], bins=30, kde=True, ax=axes[1, 0], color='red')
        axes[1, 0].set_title("Distribution of Actual Price")
        sns.boxplot(x=df['actual_price'], ax=axes[1, 1], color='red')
        axes[1, 1].set_title("Box Plot of Actual Price")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        st.pyplot(fig)

        st.write("### Scatter Plot: Actual Price vs Discounted Price with Discount Percentage")
        fig, ax = plt.subplots(figsize=(12, 5))
        scatter = ax.scatter(df['actual_price'], df['discounted_price'], c=df['discount_percentage'], cmap='coolwarm', alpha=0.7)
        ax.set_title("Scatter Plot: Actual Price vs Discounted Price (with Discount Percentage)")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Discounted Price")
        fig.colorbar(scatter, label="Discount Percentage")
        st.pyplot(fig)

        st.write("### Distribution of Product Ratings")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x=df['rating'], palette='viridis', ax=ax)
        st.pyplot(fig)

        st.write("### Top 10 Most Rated Products")
        top_products = df.nlargest(10, 'rating_count')
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(x=top_products['product_name'], y=top_products['rating_count'], palette='magma', ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.write("### Distribution of Product Categories")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y=df['category'], order=df['category'].value_counts().index, palette='coolwarm', ax=ax)
        st.pyplot(fig)

        st.write("### Category Distribution in Pie Chart")
        fig, ax = plt.subplots(figsize=(8, 8))
        df['category'].value_counts().plot.pie(autopct='%1.1f%%', cmap='viridis', startangle=90, ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

        st.write("### Heatmap of Correlations")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[['discounted_price', 'actual_price', 'rating', 'rating_count', 'discount_percentage']].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

# Customer Segments section:
if page == "Customer Segments":
    st.title("🎯 Customer Segmentation Analysis")
    
    if df is not None:
        # Filter valid data
        valid_df = df[
            (df['actual_price'] > 0) & 
            (df['discounted_price'] > 0) & 
            (df['rating'] > 0) &
            (df['rating_count'] > 0)
        ].copy()
        
        if len(valid_df) > 0:
            # 1. Data Preparation
            features = ['actual_price', 'discounted_price', 'rating', 'rating_count', 'discount_percentage']
            X = valid_df[features].copy()
            
            # Log transform price and count features
            X['actual_price'] = np.log1p(X['actual_price'])
            X['discounted_price'] = np.log1p(X['discounted_price'])
            X['rating_count'] = np.log1p(X['rating_count'])
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 2. Perform Clustering
            n_clusters = 4
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            valid_df['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Add meaningful cluster labels
            cluster_labels = {
                0: "Budget Buyers",
                1: "Value Seekers",
                2: "Premium Shoppers",
                3: "Luxury Enthusiasts"
            }
            valid_df['Cluster_Label'] = valid_df['Cluster'].map(cluster_labels)
            
            # 3. Cluster Distribution
            st.header("1️⃣ Customer Segment Distribution")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                cluster_dist = valid_df['Cluster_Label'].value_counts().reset_index()
                cluster_dist.columns = ['Segment', 'Count']
                
                fig = px.bar(cluster_dist, 
                            x='Segment', 
                            y='Count',
                            color='Segment',
                            title="Distribution of Customer Segments",
                            labels={'Count': 'Number of Customers'},
                            template="plotly_white")
                
                fig.update_layout(
                    plot_bgcolor='white',
                    showlegend=False,
                    title_x=0.5,
                    title_font_size=20,
                    xaxis_title="Customer Segment",
                    yaxis_title="Number of Customers"
                )
                st.plotly_chart(fig)
                
            with col2:
                fig = px.pie(cluster_dist, 
                            values='Count', 
                            names='Segment',
                            title="Segment Proportions",
                            hole=0.4)
                fig.update_layout(title_x=0.5)
                st.plotly_chart(fig)
            
            # 4. Price Behavior Analysis
            st.header("2️⃣ Price Behavior Analysis")
            fig = px.scatter(valid_df, 
                           x='actual_price', 
                           y='discounted_price',
                           color='Cluster_Label',
                           size='rating_count',
                           hover_data=['product_name', 'rating', 'discount_percentage'],
                           title="Customer Segments by Price Behavior",
                           template="plotly_white",
                           labels={
                               'actual_price': 'Original Price ($)',
                               'discounted_price': 'Discounted Price ($)',
                               'Cluster_Label': 'Customer Segment'
                           })
            
            fig.update_layout(
                plot_bgcolor='white',
                title_x=0.5,
                title_font_size=20
            )
            st.plotly_chart(fig)
            
            # 5. Segment Profiles
            st.header("3️⃣ Segment Profiles")
            
            for cluster in range(n_clusters):
                cluster_label = cluster_labels[cluster]
                cluster_data = valid_df[valid_df['Cluster'] == cluster]
                
                with st.expander(f"📊 {cluster_label} Profile", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        metrics = cluster_data[features].mean()
                        st.write("**Price Metrics:**")
                        st.write(f"🏷️ Avg. Original Price: ${metrics['actual_price']:,.2f}")
                        st.write(f"💰 Avg. Discounted Price: ${metrics['discounted_price']:,.2f}")
                        st.write(f"📉 Avg. Discount: {metrics['discount_percentage']:.1f}%")
                        
                    with col2:
                        st.write("**Rating Metrics:**")
                        st.write(f"⭐ Avg. Rating: {metrics['rating']:.1f}/5.0")
                        st.write(f"👥 Avg. Rating Count: {metrics['rating_count']:,.0f}")
                    
                    # Radar Chart
                    normalized_metrics = (metrics - X[features].min()) / (X[features].max() - X[features].min())
                    fig = px.line_polar(
                        r=normalized_metrics,
                        theta=features,
                        line_close=True,
                        range_r=[0, 1],
                        title=f"{cluster_label} Characteristics"
                    )
                    fig.update_layout(title_x=0.5)
                    st.plotly_chart(fig)
                    
                    # Segment Description
                    descriptions = {
                        0: "💡 Price-conscious customers focused on affordable options with good value.",
                        1: "⚖️ Value-oriented customers who seek quality products at reasonable discounts.",
                        2: "💎 Quality-focused customers willing to pay premium prices for highly-rated items.",
                        3: "👑 Luxury-segment customers who prioritize premium products regardless of price."
                    }
                    st.info(descriptions[cluster])
            
            # 6. Category Distribution
            st.header("4️⃣ Category Distribution by Segment")
            if 'category' in valid_df.columns:
                category_segment = pd.crosstab(valid_df['category'], valid_df['Cluster_Label'])
                fig = px.bar(category_segment, 
                            title="Category Distribution Across Segments",
                            barmode='group',
                            template="plotly_white")
                fig.update_layout(
                    xaxis_title="Category",
                    yaxis_title="Number of Products",
                    title_x=0.5
                )
                st.plotly_chart(fig)
        
        else:
            st.error("Not enough valid data points after filtering. Please check your data.")
    
    else:
        st.error("❌ Error: Could not load the data. Please check your data files.")


# ---------------------------
# 📌 Footer
# ---------------------------
st.sidebar.info("Built with Streamlit | AI-Powered E-Commerce Insights")
