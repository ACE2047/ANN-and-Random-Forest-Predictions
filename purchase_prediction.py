import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path, time_based=True):
    """
    Load and prepare data for predictive analysis focused on purchase behavior.
    
    Parameters:
    file_path (str): Path to the dataset
    time_based (bool): Whether to prepare data for time series analysis
    
    Returns:
    DataFrame with prepared data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    print("Loading e-commerce data...")
    df = pd.read_csv(file_path)
    
    # Convert event_time to datetime
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # Focus only on purchase events for prediction
    purchases_df = df[df['event_type'] == 'purchase'].copy()
    
    # Extract date components
    purchases_df['date'] = purchases_df['event_time'].dt.date
    purchases_df['year'] = purchases_df['event_time'].dt.year
    purchases_df['month'] = purchases_df['event_time'].dt.month
    purchases_df['day'] = purchases_df['event_time'].dt.day
    purchases_df['hour'] = purchases_df['event_time'].dt.hour
    purchases_df['day_of_week'] = purchases_df['event_time'].dt.dayofweek
    purchases_df['is_weekend'] = purchases_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Fill missing values
    purchases_df['category_code'] = purchases_df['category_code'].fillna('unknown')
    purchases_df['brand'] = purchases_df['brand'].fillna('unknown')
    
    print(f"Prepared {len(purchases_df)} purchase records for analysis")
    
    return purchases_df

def identify_corporate_customers(df, min_purchase_value=10, min_purchases=1):
    """
    Identify likely corporate customers based on purchase patterns.
    
    Parameters:
    df (DataFrame): Prepared purchase data
    min_purchase_value (float): Minimum total purchase value to be considered a corporate customer
    min_purchases (int): Minimum number of purchases to be considered a corporate customer
    
    Returns:
    DataFrame with corporate customer analysis
    """
    print("Identifying corporate customers based on purchase patterns...")
    
    # Add extensive logging
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for potential issues in the data
    print("\nChecking data integrity:")
    print(f"Null values in price column: {df['price'].isnull().sum()}")
    print(f"Null values in user_id column: {df['user_id'].isnull().sum()}")
    
    # Aggregate purchases by user
    try:
        user_purchase_stats = df.groupby('user_id').agg(
            total_spent=('price', 'sum'),
            avg_purchase=('price', 'mean'),
            purchase_count=('price', 'count'),
            categories=('category_code', lambda x: len(set(x))),
            first_purchase=('event_time', 'min'),
            last_purchase=('event_time', 'max')
        ).reset_index()
    except Exception as e:
        print(f"Error during aggregation: {e}")
        return pd.DataFrame()  # Return empty DataFrame if aggregation fails
    
    # Calculate days between first and last purchase
    user_purchase_stats['days_active'] = (user_purchase_stats['last_purchase'] - 
                                         user_purchase_stats['first_purchase']).dt.total_seconds() / (24 * 3600)
    
    # Calculate purchase frequency (purchases per day active)
    user_purchase_stats['purchase_frequency'] = user_purchase_stats['purchase_count'] / \
                                               user_purchase_stats['days_active'].clip(lower=1)
    
    # Print distribution of key metrics before filtering
    print("\nBefore Corporate Customer Filtering:")
    print("Total unique users:", len(user_purchase_stats))
    print("Total spent distribution:")
    print(user_purchase_stats['total_spent'].describe())
    print("Purchase count distribution:")
    print(user_purchase_stats['purchase_count'].describe())
    
    # Identify potential corporate customers with very lenient criteria
    corporate_customers = user_purchase_stats[
        (user_purchase_stats['total_spent'] >= min_purchase_value) & 
        (user_purchase_stats['purchase_count'] >= min_purchases)
    ].copy()
    
    # Print detailed filtering results
    print("\nAfter Corporate Customer Filtering:")
    print(f"Identified {len(corporate_customers)} potential corporate customers")
    
    if len(corporate_customers) > 0:
        print("\nCorporate Customer Overview:")
        print(corporate_customers[['total_spent', 'purchase_count', 'categories']].describe())
    
    return corporate_customers

def cluster_customers(df, corporate_df, n_clusters=3):
    """
    Cluster customers based on their purchasing behavior.
    
    Parameters:
    df (DataFrame): Prepared purchase data
    corporate_df (DataFrame): Corporate customer data
    n_clusters (int): Number of clusters for KMeans
    
    Returns:
    DataFrame with customer clusters
    """
    print("Clustering customers based on purchasing behavior...")
    
    # Join with original data to get all purchases for corporate customers
    corporate_purchases = df[df['user_id'].isin(corporate_df['user_id'])].copy()
    
    # Calculate RFM metrics (Recency, Frequency, Monetary)
    current_date = df['event_time'].max()
    
    rfm_data = corporate_purchases.groupby('user_id').agg(
        recency=('event_time', lambda x: (current_date - x.max()).total_seconds() / 86400),  # days
        frequency=('event_time', 'count'),
        monetary=('price', 'sum')
    ).reset_index()
    
    # Scale the RFM metrics
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data[['recency', 'frequency', 'monetary']])
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_data['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Add descriptive labels based on cluster characteristics
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=['recency', 'frequency', 'monetary']
    )
    
    # Sort clusters by monetary value for labeling
    cluster_labels = cluster_centers.sort_values('monetary', ascending=False)
    cluster_labels = {
        i: f"Tier {idx+1}" 
        for idx, i in enumerate(cluster_labels.index)
    }
    
    rfm_data['customer_tier'] = rfm_data['cluster'].map(cluster_labels)
    
    # Merge with corporate_df to get full details
    clustered_customers = pd.merge(corporate_df, rfm_data[['user_id', 'cluster', 'customer_tier']], 
                                 on='user_id', how='left')
    
    print(f"Customers clustered into {n_clusters} segments")
    
    return clustered_customers, cluster_centers

def analyze_purchase_patterns_by_product(df):
    """
    Analyze purchase patterns at the product level to identify trends.
    
    Parameters:
    df (DataFrame): Prepared purchase data
    
    Returns:
    DataFrame with product purchase patterns
    """
    print("Analyzing product purchase patterns...")
    
    # Group by product_id and date to get daily purchase volumes
    product_daily = df.groupby(['product_id', 'date']).agg(
        units_sold=('price', 'count'),
        revenue=('price', 'sum')
    ).reset_index()
    
    # Convert date to datetime if it's not already
    product_daily['date'] = pd.to_datetime(product_daily['date'])
    
    # Get top selling products
    top_products = df.groupby('product_id').size().nlargest(20).index
    
    # Filter to top products for visualization
    top_product_daily = product_daily[product_daily['product_id'].isin(top_products)]
    
    # Get product details for reference
    product_details = df[['product_id', 'category_code', 'brand']].drop_duplicates()
    
    return product_daily, top_product_daily, product_details

def forecast_product_demand(product_daily, product_id, forecast_days=30):
    """
    Forecast product demand using time series analysis.
    
    Parameters:
    product_daily (DataFrame): Product purchase data by date
    product_id (int): Product ID to forecast
    forecast_days (int): Number of days to forecast
    
    Returns:
    DataFrame with forecast results
    """
    print(f"Forecasting demand for product ID {product_id}...")
    
    # Filter data for the specific product
    product_data = product_daily[product_daily['product_id'] == product_id].copy()
    
    if len(product_data) < 10:
        print(f"Insufficient data for product ID {product_id}. Need at least 10 data points.")
        return None
    
    # Set date as index and sort
    product_data = product_data.set_index('date').sort_index()
    
    # Resample to ensure all dates are covered (fill missing dates with 0)
    date_range = pd.date_range(start=product_data.index.min(), end=product_data.index.max())
    product_data = product_data.reindex(date_range, fill_value=0)
    
    # Use units_sold for forecasting
    sales_series = product_data['units_sold']
    
    try:
        # Try ARIMA forecasting
        model = ARIMA(sales_series, order=(5,1,1))
        model_fit = model.fit()
        
        # Forecast future values
        forecast_result = model_fit.forecast(steps=forecast_days)
        forecast_index = pd.date_range(start=sales_series.index[-1] + pd.Timedelta(days=1), 
                                     periods=forecast_days)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_index,
            'forecast': forecast_result,
            'lower_ci': forecast_result - 1.96 * model_fit.stderr,
            'upper_ci': forecast_result + 1.96 * model_fit.stderr
        })
        
        # Set negative forecasts to 0
        forecast_df['forecast'] = forecast_df['forecast'].clip(lower=0)
        forecast_df['lower_ci'] = forecast_df['lower_ci'].clip(lower=0)
        
        print(f"Successfully generated {forecast_days}-day forecast for product {product_id}")
        
        return sales_series, forecast_df
    
    except Exception as e:
        print(f"Error forecasting product {product_id}: {str(e)}")
        return None

def identify_seasonal_trends(df):
    """
    Identify seasonal trends in purchase behavior.
    
    Parameters:
    df (DataFrame): Prepared purchase data
    
    Returns:
    Dictionary with seasonal analysis results
    """
    print("Analyzing seasonal purchase trends...")
    
    # Aggregate purchases by date
    daily_purchases = df.groupby('date').agg(
        total_sales=('price', 'sum'),
        order_count=('price', 'count')
    ).reset_index()
    
    daily_purchases['date'] = pd.to_datetime(daily_purchases['date'])
    daily_purchases = daily_purchases.set_index('date').sort_index()
    
    # Analyze by day of week
    day_of_week_sales = df.groupby('day_of_week').agg(
        total_sales=('price', 'sum'),
        order_count=('price', 'count'),
        avg_order_value=('price', 'mean')
    ).reset_index()
    
    # Map day of week to names
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week_sales['day_name'] = day_of_week_sales['day_of_week'].apply(lambda x: days[x])
    
    # Analyze by hour of day
    hour_sales = df.groupby('hour').agg(
        total_sales=('price', 'sum'),
        order_count=('price', 'count'),
        avg_order_value=('price', 'mean')
    ).reset_index()
    
    # Analyze by month (if data spans multiple months)
    month_sales = df.groupby('month').agg(
        total_sales=('price', 'sum'),
        order_count=('price', 'count'),
        avg_order_value=('price', 'mean')
    ).reset_index()
    
    seasonal_analysis = {
        'daily_purchases': daily_purchases,
        'day_of_week_sales': day_of_week_sales,
        'hour_sales': hour_sales,
        'month_sales': month_sales
    }
    
    print("Seasonal trend analysis completed")
    
    return seasonal_analysis

def train_lstm_forecast_model(product_daily, product_id, sequence_length=10, future_days=30):
    """
    Train an LSTM model to forecast product demand.
    
    Parameters:
    product_daily (DataFrame): Product purchase data by date
    product_id (int): Product ID to forecast
    sequence_length (int): Number of previous days to use for prediction
    future_days (int): Number of days to forecast
    
    Returns:
    Trained model and forecast results
    """
    print(f"Training LSTM forecast model for product ID {product_id}...")
    
    # Filter data for the specific product
    product_data = product_daily[product_daily['product_id'] == product_id].copy()
    
    if len(product_data) < sequence_length + 10:
        print(f"Insufficient data for product ID {product_id}. Need at least {sequence_length + 10} data points.")
        return None
    
    # Set date as index and sort
    product_data = product_data.set_index('date').sort_index()
    
    # Resample to ensure all dates are covered (fill missing dates with 0)
    date_range = pd.date_range(start=product_data.index.min(), end=product_data.index.max())
    product_data = product_data.reindex(date_range, fill_value=0)
    
    # Use units_sold for forecasting
    sales_series = product_data['units_sold'].values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    sales_scaled = scaler.fit_transform(sales_series.reshape(-1, 1))
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(sales_scaled) - sequence_length):
        X.append(sales_scaled[i:i + sequence_length, 0])
        y.append(sales_scaled[i + sequence_length, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split into training and testing sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # Forecast future values
    last_sequence = sales_scaled[-sequence_length:].reshape(1, sequence_length, 1)
    forecast_scaled = []
    
    for _ in range(future_days):
        next_pred = model.predict(last_sequence, verbose=0)[0, 0]
        forecast_scaled.append(next_pred)
        last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred]], axis=1)
    
    # Inverse transform to get actual values
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
    
    # Create forecast DataFrame
    forecast_index = pd.date_range(
        start=product_data.index[-1] + pd.Timedelta(days=1),
        periods=future_days
    )
    
    forecast_df = pd.DataFrame({
        'date': forecast_index,
        'forecast': forecast.flatten()
    })
    
    # Set negative forecasts to 0
    forecast_df['forecast'] = forecast_df['forecast'].clip(lower=0)
    
    print(f"Successfully trained LSTM model and generated {future_days}-day forecast")
    
    return model, forecast_df, history

def corporate_purchase_prediction(corporate_customers, df):
    """
    Predict when corporate customers are likely to make their next purchase.
    
    Parameters:
    corporate_customers (DataFrame): Identified corporate customers
    df (DataFrame): Prepared purchase data
    
    Returns:
    DataFrame with next purchase predictions
    """
    print("Predicting next corporate purchases...")
    
    # For each corporate customer, analyze their purchase frequency
    next_purchase_predictions = []
    
    for _, customer in corporate_customers.iterrows():
        user_id = customer['user_id']
        
        # Get all purchases for this customer
        user_purchases = df[df['user_id'] == user_id].copy()
        user_purchases = user_purchases.sort_values('event_time')
        
        if len(user_purchases) < 3:
            continue
        
        # Calculate time between purchases
        user_purchases['next_purchase_time'] = user_purchases['event_time'].shift(-1)
        user_purchases['days_until_next'] = (user_purchases['next_purchase_time'] - 
                                            user_purchases['event_time']).dt.total_seconds() / (24 * 3600)
        
        # Remove the last row which has NaN for next purchase
        user_purchases = user_purchases.dropna(subset=['days_until_next'])
        
        if len(user_purchases) < 2:
            continue
        
        # Calculate average and median time between purchases
        avg_days_between = user_purchases['days_until_next'].mean()
        median_days_between = user_purchases['days_until_next'].median()
        
        # Get the date of the last purchase
        last_purchase_date = user_purchases['event_time'].max()
        
        # Predict next purchase date using both average and median
        predicted_next_avg = last_purchase_date + pd.Timedelta(days=avg_days_between)
        predicted_next_median = last_purchase_date + pd.Timedelta(days=median_days_between)
        
        # Calculate average purchase amount
        avg_purchase_amount = user_purchases['price'].mean()
        
        # Most frequently purchased categories and products
        top_category = user_purchases['category_code'].value_counts().index[0]
        top_product = user_purchases['product_id'].value_counts().index[0]
        
        next_purchase_predictions.append({
            'user_id': user_id,
            'customer_tier': customer.get('customer_tier', 'Unknown'),
            'last_purchase_date': last_purchase_date,
            'avg_days_between_purchases': avg_days_between,
            'median_days_between_purchases': median_days_between,
            'predicted_next_purchase_avg': predicted_next_avg,
            'predicted_next_purchase_median': predicted_next_median,
            'days_until_next_purchase_avg': (predicted_next_avg - pd.Timestamp.now()).days,
            'days_until_next_purchase_median': (predicted_next_median - pd.Timestamp.now()).days,
            'expected_purchase_amount': avg_purchase_amount,
            'most_frequent_category': top_category,
            'most_frequent_product': top_product
        })
    
    predictions_df = pd.DataFrame(next_purchase_predictions)
    
    # Sort by imminent purchases
    predictions_df = predictions_df.sort_values('days_until_next_purchase_median')
    
    print(f"Generated next purchase predictions for {len(predictions_df)} corporate customers")
    
    return predictions_df

def visualize_corporate_predictions(predictions_df):
    """
    Visualize corporate customer purchase predictions.
    
    Parameters:
    predictions_df (DataFrame): Corporate purchase predictions
    """
    # Visualize distribution of days until next purchase
    plt.figure(figsize=(12, 6))
    sns.histplot(predictions_df['days_until_next_purchase_median'], bins=30)
    plt.title('Distribution of Days Until Next Predicted Corporate Purchase')
    plt.xlabel('Days Until Next Purchase')
    plt.axvline(x=7, color='r', linestyle='--', label='7 Days')
    plt.axvline(x=30, color='g', linestyle='--', label='30 Days')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Visualize expected purchase amounts by customer tier
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='customer_tier', y='expected_purchase_amount', data=predictions_df)
    plt.title('Expected Purchase Amount by Customer Tier')
    plt.ylabel('Expected Purchase Amount')
    plt.xlabel('Customer Tier')
    plt.tight_layout()
    plt.show()
    
    # Visualize purchases by day of week
    if 'day_of_week' in predictions_df.columns:
        plt.figure(figsize=(10, 6))
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = predictions_df['day_of_week'].value_counts().reindex(day_order)
        sns.barplot(x=day_counts.index, y=day_counts.values)
        plt.title('Corporate Purchases by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Purchases')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def visualize_product_forecasts(sales_series, forecast_df, product_id, product_details=None):
    """
    Visualize product demand forecasts.
    
    Parameters:
    sales_series (Series): Historical sales data
    forecast_df (DataFrame): Forecast results
    product_id (int): Product ID
    product_details (DataFrame): Product details for reference
    """
    # Get product details if available
    product_name = f"Product ID: {product_id}"
    if product_details is not None:
        product_info = product_details[product_details['product_id'] == product_id]
        if not product_info.empty:
            category = product_info['category_code'].values[0]
            brand = product_info['brand'].values[0]
            product_name = f"{brand} - {category} (ID: {product_id})"
    
    # Plot historical data and forecast
    plt.figure(figsize=(15, 6))
    
    # Historical data
    plt.plot(sales_series.index, sales_series.values, label='Historical Sales', color='blue')
    
    # Forecast
    plt.plot(forecast_df['date'], forecast_df['forecast'], label='Forecast', color='red')
    
    # Confidence intervals if available
    if 'lower_ci' in forecast_df.columns and 'upper_ci' in forecast_df.columns:
        plt.fill_between(
            forecast_df['date'],
            forecast_df['lower_ci'],
            forecast_df['upper_ci'],
            color='red',
            alpha=0.2,
            label='95% Confidence Interval'
        )
    
    plt.title(f'Sales Forecast for {product_name}')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generate_prediction_report(seasonal_analysis, corporate_predictions, product_forecasts):
    """
    Generate a comprehensive prediction report.
    
    Parameters:
    seasonal_analysis (dict): Seasonal trend analysis results
    corporate_predictions (DataFrame): Corporate purchase predictions
    product_forecasts (dict): Product demand forecasts
    
    Returns:
    None, prints a detailed report
    """
    print("\n====== E-COMMERCE PREDICTION REPORT ======\n")
    
    # 1. Seasonal Trends
    print("SEASONAL PURCHASE TRENDS:")
    
    # Day of week analysis
    day_sales = seasonal_analysis['day_of_week_sales']
    best_day = day_sales.loc[day_sales['order_count'].idxmax()]
    print(f"Best day for sales: {best_day['day_name']} "
          f"({best_day['order_count']} orders, ${best_day['total_sales']:.2f} total)")
    
    # Hour analysis
    hour_sales = seasonal_analysis['hour_sales']
    best_hour = hour_sales.loc[hour_sales['order_count'].idxmax()]
    print(f"Peak shopping hour: {best_hour['hour']}:00 "
          f"({best_hour['order_count']} orders, ${best_hour['total_sales']:.2f} total)")
    
    # 2. Corporate Customer Predictions
    print("\nCORPORATE CUSTOMER PREDICTIONS:")
    
    # Imminent purchases (next 7 days)
    imminent = corporate_predictions[corporate_predictions['days_until_next_purchase_median'] <= 7]
    print(f"Corporate customers likely to purchase in the next 7 days: {len(imminent)}")
    
    # Show top 5 imminent purchases
    if not imminent.empty:
        print("\nTop 5 Imminent Corporate Purchases:")
        for i, (_, row) in enumerate(imminent.head(5).iterrows()):
            print(f"{i+1}. Customer {row['user_id']} (Tier: {row['customer_tier']})")
            print(f"   Predicted purchase date: {row['predicted_next_purchase_median'].strftime('%Y-%m-%d')}")
            print(f"   Expected amount: ${row['expected_purchase_amount']:.2f}")
            print(f"   Likely to purchase: {row['most_frequent_category']}")
    
    # 3. Product Demand Forecasts
    print("\nPRODUCT DEMAND FORECASTS:")
    
    # Show forecast summaries for each product
    for product_id, forecast_data in product_forecasts.items():
        sales_series, forecast_df = forecast_data
        
        # Calculate total forecasted sales for next 30 days
        total_forecast = forecast_df['forecast'].sum()
        
        # Calculate average daily sales (historical)
        avg_historical = sales_series.mean()
        
        # Calculate growth/decline
        pct_change = ((forecast_df['forecast'].mean() / avg_historical) - 1) * 100
        
        trend = "growth" if pct_change > 0 else "decline"
        
        print(f"\nProduct ID: {product_id}")
        print(f"Forecasted sales (next 30 days): {total_forecast:.0f} units")
        print(f"Projected trend: {abs(pct_change):.1f}% {trend} compared to historical average")
    
    print("\n======= END OF PREDICTION REPORT =======")

def main():
    file_path = 'data/ecommerce_behavior.csv'
    
    try:
        # Load and prepare data
        purchases_df = load_and_prepare_data(file_path)
        
        # Debug dataset information
        print("\n===== DATASET DEBUGGING =====")
        print(f"Total records: {len(purchases_df)}")
        
        print("\nDataset Columns:")
        print(purchases_df.columns.tolist())
        
        print("\nPurchase Price Summary:")
        print(purchases_df['price'].describe())
        
        print("\nUser Purchase Count:")
        user_purchase_counts = purchases_df['user_id'].value_counts()
        print(user_purchase_counts.head())
        
        print("\nEvent Type Distribution:")
        print(purchases_df['event_type'].value_counts())
        
        print("\nCategory Code Distribution:")
        print(purchases_df['category_code'].value_counts().head())
        
        print("\nUnique Users:")
        print(f"Total unique users: {purchases_df['user_id'].nunique()}")
        
        # Identify corporate customers with verbose debugging
        print("\n===== CORPORATE CUSTOMER IDENTIFICATION =====")
        corporate_customers = identify_corporate_customers(
            purchases_df, min_purchase_value=50, min_purchases=1
        )
        
        # Print corporate customer details if found
        if not corporate_customers.empty:
            print("\nCorporate Customer Details:")
            print(corporate_customers.head())
            print("\nCorporate Customer Columns:")
            print(corporate_customers.columns.tolist())
        
        # Continue with the rest of the analysis if corporate customers found
        if not corporate_customers.empty:
            # Cluster corporate customers
            clustered_customers, cluster_centers = cluster_customers(
                purchases_df, corporate_customers, n_clusters=3
            )
            
            # Analyze purchase patterns by product
            product_daily, top_product_daily, product_details = analyze_purchase_patterns_by_product(purchases_df)
            
            # Identify seasonal trends
            seasonal_analysis = identify_seasonal_trends(purchases_df)
            
            # Get top selling products for forecasting
            top_products = purchases_df['product_id'].value_counts().head(5).index.tolist()
            
            # Generate forecasts for top products
            product_forecasts = {}
            for product_id in top_products:
                forecast_result = forecast_product_demand(product_daily, product_id, forecast_days=30)
                if forecast_result:
                    product_forecasts[product_id] = forecast_result
            
            # Predict corporate purchases
            corporate_predictions = corporate_purchase_prediction(clustered_customers, purchases_df)
            
            # Generate prediction report
            generate_prediction_report(seasonal_analysis, corporate_predictions, product_forecasts)
            
            # Visualize some results
            print("\nGenerating visualizations...")
            
            # Corporate predictions
            visualize_corporate_predictions(corporate_predictions)
            
            # Product forecasts
            for product_id, forecast_data in product_forecasts.items():
                sales_series, forecast_df = forecast_data
                visualize_product_forecasts(sales_series, forecast_df, product_id, product_details)
            
            print("\nPrediction analysis completed successfully.")
        else:
            print("No corporate customers found. Cannot proceed with further analysis.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the file 'ecommerce_behavior.csv' is in the data directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()