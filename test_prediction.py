import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import multiprocessing

# Set number of available cores
NUM_CORES = max(1, multiprocessing.cpu_count() - 1)

warnings.filterwarnings('ignore')

def load_and_prepare_data(directory_path):
    """
    Load and prepare data from all CSV files in the specified directory.
    """
    all_purchases = []
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    for filename in csv_files:
        file_path = os.path.join(directory_path, filename)
        
        print(f"Loading data from {filename}...")
        try:
            df = pd.read_csv(file_path)
            
            # Convert event_time to datetime
            df['event_time'] = pd.to_datetime(df['event_time'])
            
            # Focus only on purchase events
            purchases_df = df[df['event_type'] == 'purchase'].copy()
            
            if not purchases_df.empty:
                all_purchases.append(purchases_df)
            else:
                print(f"No purchase events found in {filename}")
        
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    # Combine all purchase datasets
    if not all_purchases:
        raise ValueError("No purchase data found in any files")
    
    combined_purchases = pd.concat(all_purchases, ignore_index=True)
    
    # Date components extraction
    combined_purchases['date'] = combined_purchases['event_time'].dt.date
    combined_purchases['year'] = combined_purchases['event_time'].dt.year
    combined_purchases['month'] = combined_purchases['event_time'].dt.month
    combined_purchases['day'] = combined_purchases['event_time'].dt.day
    combined_purchases['hour'] = combined_purchases['event_time'].dt.hour
    combined_purchases['day_of_week'] = combined_purchases['event_time'].dt.dayofweek
    combined_purchases['is_weekend'] = combined_purchases['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    combined_purchases['quarter'] = combined_purchases['event_time'].dt.quarter  # Add quarter for seasonality
    
    # Add holiday indicator (example: December is holiday season)
    combined_purchases['is_holiday_season'] = combined_purchases['month'].apply(lambda x: 1 if x == 12 else 0)
    
    # Fill missing values
    combined_purchases['category_code'] = combined_purchases['category_code'].fillna('unknown')
    combined_purchases['brand'] = combined_purchases['brand'].fillna('unknown')
    
    print(f"Prepared {len(combined_purchases)} total purchase records for analysis")
    
    return combined_purchases

def identify_high_value_customers(df, min_total_spend=300, min_purchases=2):
    """
    Enhanced high-value customer identification with vectorized operations.
    """
    print("Identifying high-value customers across expanded dataset...")
    
    # Aggregate purchases by user - use vectorized operations
    user_stats = df.groupby('user_id').agg(
        total_spent=('price', 'sum'),
        unique_categories=('category_code', 'nunique'),
        purchase_count=('price', 'count'),
        first_purchase=('event_time', 'min'),
        last_purchase=('event_time', 'max'),
        avg_purchase_value=('price', 'mean')
    ).reset_index()
    
    # Calculate days active - vectorized
    user_stats['days_active'] = (user_stats['last_purchase'] - user_stats['first_purchase']).dt.days
    user_stats['purchase_frequency'] = user_stats['purchase_count'] / (user_stats['days_active'] + 1)  # Add 1 to avoid division by zero
    
    # Identify purchase recency (days since last purchase) - vectorized
    user_stats['days_since_last_purchase'] = (pd.Timestamp.now(tz='UTC') - user_stats['last_purchase']).dt.days
    
    # RFM Score (Recency, Frequency, Monetary)
    # Convert metrics to quintiles (1-5 score)
    user_stats['recency_score'] = pd.qcut(user_stats['days_since_last_purchase'], 5, labels=False, duplicates='drop')
    user_stats['recency_score'] = 5 - user_stats['recency_score']  # Reverse (lower days = higher score)
    user_stats['frequency_score'] = pd.qcut(user_stats['purchase_frequency'].clip(lower=0), 5, labels=False, duplicates='drop')
    user_stats['monetary_score'] = pd.qcut(user_stats['total_spent'], 5, labels=False, duplicates='drop')
    
    # Calculate combined RFM score - vectorized
    user_stats['rfm_score'] = user_stats['recency_score'] + user_stats['frequency_score'] + user_stats['monetary_score']
    
    # Comprehensive customer segmentation - vectorized approach
    conditions = [
        user_stats['rfm_score'] >= 12,
        user_stats['rfm_score'] >= 9,
        user_stats['rfm_score'] >= 6,
        user_stats['rfm_score'] >= 0
    ]
    choices = ['Premium', 'High-Value', 'Regular', 'Occasional']
    user_stats['customer_segment'] = np.select(conditions, choices, default='Occasional')
    
    # Diagnostic information
    print("\n=== Comprehensive Customer Analysis ===")
    print("Customer Segment Distribution:")
    print(user_stats['customer_segment'].value_counts())
    
    print("\nCustomer Segment Statistics:")
    segment_stats = user_stats.groupby('customer_segment').agg({
        'total_spent': ['mean', 'max'],
        'purchase_count': ['mean', 'max'],
        'days_active': ['mean', 'max'],
        'rfm_score': ['mean', 'min', 'max']
    })
    print(segment_stats)
    
    # Filter high-value customers
    high_value_customers = user_stats[
        (user_stats['customer_segment'].isin(['Premium', 'High-Value'])) |
        ((user_stats['total_spent'] >= min_total_spend) & (user_stats['purchase_count'] >= min_purchases))
    ].copy()
    
    print(f"\nHigh-value customers found: {len(high_value_customers)}")
    
    # Display top high-value customers
    top_customers = high_value_customers.sort_values('total_spent', ascending=False).head(15)
    print("\nTop 15 High-Value Customers:")
    print(top_customers[['user_id', 'total_spent', 'purchase_count', 'customer_segment', 'rfm_score']])
    
    return high_value_customers

def extract_customer_features(df):
    """
    Extract features for customer clustering.
    """
    # Extract behavioral features for clustering
    features = []
    
    for user_id, user_data in df.groupby('user_id'):
        # Basic purchase statistics
        total_spent = user_data['price'].sum()
        purchase_count = len(user_data)
        avg_purchase = user_data['price'].mean()
        
        # Temporal patterns
        date_range = (user_data['event_time'].max() - user_data['event_time'].min()).total_seconds() / (3600 * 24)
        purchase_frequency = purchase_count / max(1, date_range)
        
        # Time of day preferences (morning, afternoon, evening, night)
        hour_dist = user_data['hour'].value_counts(normalize=True).to_dict()
        morning_pct = sum(hour_dist.get(h, 0) for h in range(5, 12))
        afternoon_pct = sum(hour_dist.get(h, 0) for h in range(12, 17))
        evening_pct = sum(hour_dist.get(h, 0) for h in range(17, 22))
        night_pct = sum(hour_dist.get(h, 0) for h in range(22, 24)) + sum(hour_dist.get(h, 0) for h in range(0, 5))
        
        # Day of week preferences
        weekday_pct = sum(user_data['is_weekend'] == 0) / len(user_data)
        weekend_pct = sum(user_data['is_weekend'] == 1) / len(user_data)
        
        # Category diversity
        category_diversity = user_data['category_code'].nunique() / purchase_count
        
        # Feature vector
        features.append({
            'user_id': user_id,
            'total_spent': total_spent,
            'purchase_count': purchase_count,
            'avg_purchase': avg_purchase,
            'purchase_frequency': purchase_frequency,
            'morning_pct': morning_pct,
            'afternoon_pct': afternoon_pct,
            'evening_pct': evening_pct,
            'night_pct': night_pct,
            'weekday_pct': weekday_pct,
            'weekend_pct': weekend_pct,
            'category_diversity': category_diversity
        })
    
    return pd.DataFrame(features).set_index('user_id')

def cluster_customers(features_df, n_clusters=5):
    """
    Cluster customers based on behavioral features.
    """
    print(f"Clustering customers into {n_clusters} behavior groups...")
    
    # Select features for clustering
    clustering_features = [
        'total_spent', 'purchase_count', 'avg_purchase', 'purchase_frequency',
        'morning_pct', 'afternoon_pct', 'evening_pct', 'night_pct',
        'weekday_pct', 'weekend_pct', 'category_diversity'
    ]
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df[clustering_features])
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to features
    features_df['cluster'] = clusters
    
    # Display cluster information
    print("\nCustomer Clusters Distribution:")
    print(features_df['cluster'].value_counts())
    
    print("\nCluster Characteristics:")
    cluster_profiles = features_df.groupby('cluster')[clustering_features].mean()
    print(cluster_profiles)
    
    return features_df

def prepare_time_series_data(df, user_ids):
    """
    Prepare time series data for a group of customers.
    """
    all_user_purchases = df[df['user_id'].isin(user_ids)].copy()
    
    # Create daily purchase aggregation by user
    all_user_purchases['date'] = all_user_purchases['event_time'].dt.date
    daily_purchases = all_user_purchases.groupby(['user_id', 'date']).agg(
        daily_spend=('price', 'sum'),
        purchase_count=('price', 'count')
    ).reset_index()
    
    # Convert to time series with proper date index
    daily_purchases['date'] = pd.to_datetime(daily_purchases['date'])
    
    return daily_purchases

def train_cluster_arima_model(time_series_data, train_size=0.8):
    """
    Train ARIMA model on clustered time series data.
    """
    # Aggregate by date across all users in cluster
    cluster_ts = time_series_data.groupby('date')['daily_spend'].sum().sort_index()
    
    # Split into train and test sets
    split_idx = int(len(cluster_ts) * train_size)
    if split_idx < 10:  # Need minimum data
        return None, None, None, None
    
    train_data = cluster_ts.iloc[:split_idx]
    test_data = cluster_ts.iloc[split_idx:]
    
    if len(train_data) < 10 or len(test_data) < 5:  # Minimum data requirements
        return None, None, None, None
    
    best_mae = float('inf')
    best_model = None
    best_order = None
    
    # Try different ARIMA parameters
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    model_fit = model.fit()
                    
                    # Forecast for test period
                    forecast = model_fit.forecast(steps=len(test_data))
                    
                    # Calculate error
                    mae = mean_absolute_error(test_data, forecast)
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_model = model_fit
                        best_order = (p, d, q)
                        
                except Exception as e:
                    continue
    
    if best_model is None:
        return None, None, None, None
        
    # Calculate prediction accuracy on test set
    forecast = best_model.forecast(steps=len(test_data))
    
    # Binary accuracy (did we correctly predict purchase vs. no purchase?)
    actual_binary = (test_data > 0).astype(int)
    forecast_binary = (forecast > 0).astype(int)
    
    binary_accuracy = (actual_binary == forecast_binary).mean() * 100
    
    return best_model, best_order, binary_accuracy, best_mae

def process_cluster(cluster_id, cluster_users, df, prediction_days=30):
    """
    Process one cluster of users in parallel.
    """
    print(f"Processing cluster {cluster_id} with {len(cluster_users)} users...")
    
    # Prepare time series data for this cluster
    cluster_time_series = prepare_time_series_data(df, cluster_users)
    
    # Train cluster model
    model, order, accuracy, mae = train_cluster_arima_model(cluster_time_series)
    
    if model is None:
        print(f"Could not train model for cluster {cluster_id}. Insufficient data.")
        return cluster_id, None, None
    
    print(f"Cluster {cluster_id} model: ARIMA{order}, Accuracy: {accuracy:.2f}%, MAE: {mae:.2f}")
    
    # Generate predictions for each user in the cluster
    predictions = []
    
    for user_id in cluster_users:
        user_purchases = df[df['user_id'] == user_id].sort_values('event_time')
        
        if len(user_purchases) < 2:
            continue
            
        # Get last purchase date
        last_purchase = user_purchases['event_time'].max()
        
        # Forecast using cluster model
        forecast = model.forecast(steps=prediction_days)
        
        # Find first day with predicted purchase
        purchase_days = np.where(forecast > 0)[0]
        if len(purchase_days) > 0:
            days_until_purchase = purchase_days[0]
        else:
            days_until_purchase = prediction_days  # Default to max prediction window
        
        next_prediction = last_purchase + pd.Timedelta(days=days_until_purchase)
        
        # Use cluster model accuracy as confidence
        confidence_score = min(1.0, accuracy / 100) if accuracy is not None else 0.6
        
        predictions.append({
            'user_id': user_id,
            'cluster_id': cluster_id,
            'prediction_method': 'Cluster_ARIMA',
            'arima_order': str(order),
            'predicted_next_purchase': next_prediction,
            'confidence_score': confidence_score,
            'cluster_accuracy': accuracy,
            'cluster_mae': mae
        })
    
    return cluster_id, predictions, accuracy

def enhanced_purchase_prediction_with_clustering(high_value_customers, df, prediction_days=30):
    """
    Enhanced purchase prediction using customer clustering and parallel processing.
    """
    print("\n=== Enhanced Purchase Prediction with Clustering and Parallel Processing ===")
    
    # Step 1: Extract customer features for clustering
    print("Extracting customer behavioral features...")
    customer_features = extract_customer_features(df)
    
    # Join with high value customers to get segment info
    merged_features = pd.merge(
        customer_features,
        high_value_customers[['user_id', 'customer_segment', 'rfm_score']],
        left_index=True, right_on='user_id'
    ).set_index('user_id')
    
    # Step 2: Cluster customers by behavior patterns
    optimal_clusters = min(5, len(merged_features) // 20)  # At least 20 customers per cluster
    optimal_clusters = max(2, optimal_clusters)  # At least 2 clusters
    
    clustered_customers = cluster_customers(merged_features, n_clusters=optimal_clusters)
    
    # Step 3: Process each cluster in parallel
    clusters = clustered_customers['cluster'].unique()
    
    # Prepare for parallel processing
    print(f"\nProcessing {len(clusters)} clusters in parallel using {NUM_CORES} cores...")
    
    cluster_results = Parallel(n_jobs=NUM_CORES)(
        delayed(process_cluster)(
            cluster_id,
            clustered_customers[clustered_customers['cluster'] == cluster_id].index,
            df,
            prediction_days
        ) for cluster_id in clusters
    )
    
    # Combine all predictions
    all_predictions = []
    cluster_accuracies = {}
    
    for cluster_id, predictions, accuracy in cluster_results:
        if predictions:
            all_predictions.extend(predictions)
        if accuracy:
            cluster_accuracies[cluster_id] = accuracy
    
    # Create DataFrame from predictions
    predictions_df = pd.DataFrame(all_predictions)
    
    # Report on results
    if cluster_accuracies:
        print("\nCluster Model Accuracies:")
        for cluster_id, accuracy in cluster_accuracies.items():
            print(f"Cluster {cluster_id}: {accuracy:.2f}%")
        
        print(f"\nOverall average cluster accuracy: {np.mean(list(cluster_accuracies.values())):.2f}%")
    
    print(f"\nTotal predictions generated: {len(predictions_df)}")
    
    return predictions_df.sort_values('predicted_next_purchase')

def calculate_prediction_window_accuracy(predictions_df, purchases_df, window_days=7):
    """
    Calculate accuracy by checking if purchases occurred within the predicted window.
    """
    results = []
    
    for _, prediction in predictions_df.iterrows():
        user_id = prediction['user_id']
        predicted_date = prediction['predicted_next_purchase']
        
        # Get actual purchases after prediction date
        user_purchases = purchases_df[
            (purchases_df['user_id'] == user_id) & 
            (purchases_df['event_time'] > predicted_date)
        ]
        
        if user_purchases.empty:
            # No purchases found in data period (can't determine accuracy)
            continue
        
        # Get earliest purchase after prediction
        actual_next_purchase = user_purchases['event_time'].min()
        
        # Calculate days difference
        days_diff = (actual_next_purchase - predicted_date).total_seconds() / (24 * 3600)
        
        # Check if purchase occurred within window
        accurate = abs(days_diff) <= window_days
        
        results.append({
            'user_id': user_id,
            'predicted_date': predicted_date,
            'actual_date': actual_next_purchase,
            'days_difference': days_diff,
            'accurate_within_window': accurate
        })
    
    if not results:
        print("No validation data available - predictions may be for future dates")
        return None
    
    results_df = pd.DataFrame(results)
    
    # Calculate overall window accuracy
    window_accuracy = results_df['accurate_within_window'].mean() * 100
    
    print(f"\nAccuracy within {window_days}-day window: {window_accuracy:.2f}%")
    print(f"Average days difference: {results_df['days_difference'].abs().mean():.2f}")
    
    return results_df

def visualize_prediction_accuracy(validation_results):
    """
    Create visualizations for prediction accuracy.
    """
    if validation_results is None or len(validation_results) < 5:
        print("Insufficient validation data for visualization")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Days difference distribution
    plt.subplot(1, 2, 1)
    plt.hist(validation_results['days_difference'], bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Prediction Error Distribution (Days)')
    plt.xlabel('Days Difference (Negative = Early, Positive = Late)')
    plt.ylabel('Count')
    
    # Accuracy by day range
    plt.subplot(1, 2, 2)
    day_ranges = [1, 3, 5, 7, 14, 30]
    accuracies = []
    
    for days in day_ranges:
        accuracy = (abs(validation_results['days_difference']) <= days).mean() * 100
        accuracies.append(accuracy)
    
    plt.bar(range(len(day_ranges)), accuracies, color='lightgreen')
    plt.xticks(range(len(day_ranges)), [f"±{d} days" for d in day_ranges])
    plt.title('Prediction Accuracy by Time Window')
    plt.xlabel('Time Window')
    plt.ylabel('Accuracy (%)')
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('prediction_accuracy.png')
    print("Saved accuracy visualization to 'prediction_accuracy.png'")

def main():
    # Specify the directory containing CSV files
    directory_path = 'data'
    
    try:
        # Load and prepare comprehensive data
        purchases_df = load_and_prepare_data(directory_path)
        
        # Time-based train/test split (for validation)
        # Use 80% of data for training, 20% for testing
        cutoff_date = purchases_df['event_time'].quantile(0.8)
        training_data = purchases_df[purchases_df['event_time'] <= cutoff_date].copy()
        validation_data = purchases_df[purchases_df['event_time'] > cutoff_date].copy()
        
        print(f"\nData split at {cutoff_date}")
        print(f"Training data: {len(training_data)} records ({training_data['event_time'].min()} to {training_data['event_time'].max()})")
        print(f"Validation data: {len(validation_data)} records ({validation_data['event_time'].min()} to {validation_data['event_time'].max()})")
        
        # Identify high-value customers using training data
        high_value_customers = identify_high_value_customers(
            training_data, min_total_spend=300, min_purchases=2
        )
        
        # Check if any high-value customers were found
        if len(high_value_customers) == 0:
            print("No high-value customers found. Adjust criteria or check data.")
            return
        
        # Use the new clustered prediction approach
        purchase_predictions = enhanced_purchase_prediction_with_clustering(
            high_value_customers, training_data, prediction_days=30
        )
        
        # Validate predictions using holdout data
        validation_results = calculate_prediction_window_accuracy(
            purchase_predictions, validation_data, window_days=7
        )
        
        # Create visualizations
        visualize_prediction_accuracy(validation_results)
        
        # Save final predictions and validation results
        purchase_predictions.to_csv('purchase_predictions.csv', index=False)
        if validation_results is not None:
            validation_results.to_csv('prediction_validation.csv', index=False)
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()