import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="E-commerce Analysis Dashboard", page_icon="💎", layout="wide")

@st.cache_data
def load_data():
    orders_df = pd.read_csv('data/orders_dataset.csv')
    order_items_df = pd.read_csv('data/order_items_dataset.csv')
    customers_df = pd.read_csv('data/customers_dataset.csv')
    return orders_df, order_items_df, customers_df

orders_df, order_items_df, customers_df = load_data()

st.title('E-commerce Analysis Dashboard 💎')

tabs = st.tabs(['Data Overview', 'EDA', 'RFM Analysis'])

with tabs[0]:
    st.header('Data Overview')

    st.subheader('Orders Dataset')
    st.dataframe(orders_df.head())
    st.write('Shape:', orders_df.shape)

    st.subheader('Order Items Dataset')
    st.dataframe(order_items_df.head())
    st.write('Shape:', order_items_df.shape)

    st.subheader('Customers Dataset')
    st.dataframe(customers_df.head())
    st.write('Shape:', customers_df.shape)

with tabs[1]:
    st.header('Exploratory Data Analysis')

    # Numeric Analysis
    st.subheader('Numeric Variables Distribution')
    numeric_cols = order_items_df.select_dtypes(include=['float64', 'int64']).columns

    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(15, 5))
    for i, col in enumerate(numeric_cols):
        sns.histplot(data=order_items_df, x=col, ax=axes[i])
        axes[i].set_title(f'{col} Distribution')
    plt.tight_layout()
    st.pyplot(fig)

    # Order Status Analysis
    st.subheader('Order Status Distribution')
    status_counts = orders_df['order_status'].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    status_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Order Status Counts')

    status_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
    ax2.set_title('Order Status Distribution')
    plt.tight_layout()
    st.pyplot(fig)

    # Geographic Analysis
    st.subheader('Geographic Distribution')
    state_counts = customers_df['customer_state'].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    state_counts.plot(kind='bar')
    plt.title('Customers by State')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Time Series Analysis
    st.subheader('Order Trends')
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    orders_df['month_year'] = orders_df['order_purchase_timestamp'].dt.to_period('M')
    monthly_orders = orders_df.groupby('month_year').size()

    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_orders.plot(kind='line')
    plt.title('Monthly Orders Trend')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Price Analysis
    st.subheader('Price Analysis')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    sns.boxplot(data=order_items_df, y='price', ax=ax1)
    ax1.set_title('Price Distribution')

    sns.scatterplot(data=order_items_df, x='price', y='freight_value', ax=ax2)
    ax2.set_title('Price vs Freight Value')
    plt.tight_layout()
    st.pyplot(fig)

with tabs[2]:
    st.header('RFM Analysis')

    @st.cache_data
    def calculate_rfm():
        # RFM calculation code (same as before)
        date_columns = ['order_purchase_timestamp', 'order_delivered_customer_date']
        for col in date_columns:
            orders_df[col] = pd.to_datetime(orders_df[col])

        orders_clean = orders_df[
            (orders_df['order_status'] == 'delivered') &
            (orders_df['order_delivered_customer_date'].notna())
        ]

        order_values = order_items_df.groupby('order_id')['price'].sum().reset_index()
        orders_clean = orders_clean.merge(order_values, on='order_id', how='left')

        current_date = orders_clean['order_delivered_customer_date'].max()

        rfm_df = orders_clean.groupby('customer_id').agg({
            'order_delivered_customer_date': lambda x: (current_date - x.max()).days,
            'order_id': 'count',
            'price': 'sum'
        }).reset_index()

        rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary']

        rfm_df['R'] = pd.qcut(rfm_df['recency'], q=4, labels=['4','3','2','1'])
        rfm_df['F'] = 1
        rfm_df['M'] = pd.qcut(rfm_df['monetary'], q=4, labels=['1','2','3','4'])

        return rfm_df

    rfm_data = calculate_rfm()

    rfm_tabs = st.tabs(['Overview', 'Brand Ambassadors', 'Retention Targets'])

    with rfm_tabs[0]:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Customers", f"{len(rfm_data):,}")
        with col2:
            st.metric("Average Order Value", f"R$ {rfm_data['monetary'].mean():.2f}")
        with col3:
            st.metric("Average Recency (days)", f"{rfm_data['recency'].mean():.0f}")

        st.dataframe(rfm_data.describe())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(data=rfm_data, x='recency', ax=ax1)
        sns.histplot(data=rfm_data, x='monetary', ax=ax2)
        plt.tight_layout()
        st.pyplot(fig)

    with rfm_tabs[1]:
        ambassadors = rfm_data[(rfm_data['R'] == '4') & (rfm_data['M'] == '4')]

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Potential Ambassadors",
                f"{len(ambassadors):,}",
                f"{(len(ambassadors)/len(rfm_data)*100):.1f}% of customers"
            )
        with col2:
            st.metric(
                "Average Ambassador Spend",
                f"R$ {ambassadors['monetary'].mean():.2f}",
                f"{(ambassadors['monetary'].mean()/rfm_data['monetary'].mean()-1)*100:.1f}% vs average"
            )

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.scatter(rfm_data['recency'], rfm_data['monetary'], alpha=0.5, color='gray')
        plt.scatter(ambassadors['recency'], ambassadors['monetary'], alpha=0.7, color='red', label='Potential Ambassadors')
        plt.xlabel('Recency (days)')
        plt.ylabel('Monetary Value (R$)')
        plt.title('Brand Ambassadors Identification')
        plt.legend()
        st.pyplot(fig)

    with rfm_tabs[2]:
        retention = rfm_data[(rfm_data['R'] == '1') & (rfm_data['M'] == '4')]

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Retention Targets",
                f"{len(retention):,}",
                f"{(len(retention)/len(rfm_data)*100):.1f}% of customers"
            )
        with col2:
            st.metric(
                "Revenue at Risk",
                f"R$ {retention['monetary'].sum():,.2f}",
                "Needs attention",
                delta_color="inverse"
            )

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.hist(retention['recency'], bins=30)
        plt.xlabel('Days Since Last Purchase')
        plt.ylabel('Number of Customers')
        plt.title('Retention Targets - Time Since Last Purchase')
        st.pyplot(fig)

st.markdown('---')
st.caption('Created for Dicoding Data Analysis Project')