import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

def load_and_analyze_data(file_path):
    """Load and perform detailed analysis of the e-commerce behavior dataset."""
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: Could not find the dataset at {file_path}")
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    # Load the data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    # Convert event_time to datetime
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # Extract date components
    df['date'] = df['event_time'].dt.date
    df['hour'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Basic information about the dataset
    print("\n=== Dataset Overview ===")
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    
    # Check for missing values
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    # Basic statistics for numerical columns
    print("\n=== Numerical Features Statistics ===")
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print(df[numerical_cols].describe())
    
    # Categorical columns overview
    print("\n=== Categorical Features Overview ===")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col} - Unique Values: {df[col].nunique()}")
        if df[col].nunique() < 20:  # Only show distribution for columns with fewer unique values
            print(df[col].value_counts(normalize=True).head(10))
    
    return df

def plot_data_distributions(df):
    """Create visualizations for data distributions."""
    # Set up the style
    plt.style.use('ggplot')
    
    # Analyze event types
    plt.figure(figsize=(10, 6))
    event_counts = df['event_type'].value_counts()
    sns.barplot(x=event_counts.index, y=event_counts.values)
    plt.title('Distribution of Event Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Price distribution
    plt.figure(figsize=(10, 6))
    # Filter out any extreme outliers for better visualization
    price_filtered = df['price'].dropna()
    price_filtered = price_filtered[price_filtered < price_filtered.quantile(0.95)]
    sns.histplot(price_filtered, bins=30)
    plt.title('Price Distribution (Excluding Outliers)')
    plt.tight_layout()
    plt.show()
    
    # Events by hour of day
    plt.figure(figsize=(12, 6))
    hourly_events = df.groupby('hour').size()
    sns.barplot(x=hourly_events.index, y=hourly_events.values)
    plt.title('Events by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Events')
    plt.tight_layout()
    plt.show()
    
    # Events by day of week
    plt.figure(figsize=(10, 6))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_events = df.groupby('day_of_week').size()
    sns.barplot(x=day_events.index, y=day_events.values)
    plt.xticks(ticks=range(7), labels=days, rotation=45)
    plt.title('Events by Day of Week')
    plt.xlabel('Day')
    plt.ylabel('Number of Events')
    plt.tight_layout()
    plt.show()

def plot_category_analysis(df):
    """Analyze product categories."""
    # Top categories by event count
    plt.figure(figsize=(12, 8))
    
    # Handle missing category_codes
    df['category_code_filled'] = df['category_code'].fillna('unknown')
    
    # Get top categories (excluding unknown)
    top_categories = df['category_code_filled'].value_counts()
    top_categories = top_categories[top_categories.index != 'unknown'].head(15)
    
    sns.barplot(x=top_categories.values, y=top_categories.index)
    plt.title('Top 15 Product Categories by Event Count')
    plt.xlabel('Number of Events')
    plt.tight_layout()
    plt.show()
    
    # Top brands by event count
    plt.figure(figsize=(12, 8))
    
    # Handle missing brands
    df['brand_filled'] = df['brand'].fillna('unknown')
    
    # Get top brands (excluding unknown)
    top_brands = df['brand_filled'].value_counts()
    top_brands = top_brands[top_brands.index != 'unknown'].head(15)
    
    sns.barplot(x=top_brands.values, y=top_brands.index)
    plt.title('Top 15 Brands by Event Count')
    plt.xlabel('Number of Events')
    plt.tight_layout()
    plt.show()

def analyze_purchase_patterns(df):
    """Analyze purchase events specifically."""
    # Filter for purchase events
    purchases = df[df['event_type'] == 'purchase']
    
    # Purchases by hour
    plt.figure(figsize=(12, 6))
    purchase_by_hour = purchases.groupby('hour').size()
    sns.barplot(x=purchase_by_hour.index, y=purchase_by_hour.values)
    plt.title('Purchases by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Purchases')
    plt.tight_layout()
    plt.show()
    
    # Purchases by day of week
    plt.figure(figsize=(10, 6))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    purchase_by_day = purchases.groupby('day_of_week').size()
    sns.barplot(x=purchase_by_day.index, y=purchase_by_day.values)
    plt.xticks(ticks=range(7), labels=days, rotation=45)
    plt.title('Purchases by Day of Week')
    plt.xlabel('Day')
    plt.ylabel('Number of Purchases')
    plt.tight_layout()
    plt.show()
    
    # Average purchase price by category (top 10)
    plt.figure(figsize=(12, 8))
    
    # Use filled category codes
    purchases['category_code_filled'] = purchases['category_code'].fillna('unknown')
    
    # Get average prices by category
    avg_price_by_category = purchases.groupby('category_code_filled')['price'].mean().sort_values(ascending=False)
    avg_price_by_category = avg_price_by_category[avg_price_by_category.index != 'unknown'].head(10)
    
    sns.barplot(x=avg_price_by_category.values, y=avg_price_by_category.index)
    plt.title('Top 10 Categories by Average Purchase Price')
    plt.xlabel('Average Price')
    plt.tight_layout()
    plt.show()

def generate_summary_report(df):
    """Generate a summary report of key findings."""
    print("\n=== Summary Report ===")
    
    # Event type distribution
    print("\nEvent Type Distribution:")
    event_type_dist = df['event_type'].value_counts(normalize=True)
    for event_type, percentage in event_type_dist.items():
        print(f"{event_type}: {percentage:.2%}")
    
    # Calculate conversion rate (view to purchase)
    views = df[df['event_type'] == 'view'].shape[0]
    purchases = df[df['event_type'] == 'purchase'].shape[0]
    if views > 0:
        conversion_rate = purchases / views
        print(f"\nConversion Rate (View to Purchase): {conversion_rate:.2%}")
    
    # Average price
    avg_price = df['price'].mean()
    print(f"\nAverage Product Price: ${avg_price:.2f}")
    
    # Price by event type
    price_by_event = df.groupby('event_type')['price'].mean()
    print("\nAverage Price by Event Type:")
    for event_type, avg_price in price_by_event.items():
        print(f"{event_type}: ${avg_price:.2f}")
    
    # User activity
    unique_users = df['user_id'].nunique()
    unique_sessions = df['user_session'].nunique()
    events_per_user = df.shape[0] / unique_users
    print(f"\nUnique Users: {unique_users}")
    print(f"Unique Sessions: {unique_sessions}")
    print(f"Average Events per User: {events_per_user:.2f}")
    
    # Weekend vs. Weekday
    weekend_events = df[df['is_weekend'] == 1].shape[0]
    weekday_events = df[df['is_weekend'] == 0].shape[0]
    total_events = weekend_events + weekday_events
    print(f"\nWeekend Events: {weekend_events} ({weekend_events/total_events:.2%})")
    print(f"Weekday Events: {weekday_events} ({weekday_events/total_events:.2%})")

def main():
    # Define the file path
    file_path = 'data/ecommerce_behavior.csv'
    
    try:
        # Load and analyze data
        df = load_and_analyze_data(file_path)
        
        # Generate visualizations
        plot_data_distributions(df)
        plot_category_analysis(df)
        analyze_purchase_patterns(df)
        
        # Generate summary report
        generate_summary_report(df)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the file 'ecommerce_behavior.csv' is in the data directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()