import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path, time_based=True):
    """
    Load and prepare data for predictive analysis focused on purchase behavior.
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

def identify_high_value_customers(df, min_total_spend=500, min_purchases=3):
    """
    Identify high-value customers with flexible criteria and diagnostic information.
    """
    print("Identifying high-value customers...")
    
    # Aggregate purchases by user
    user_stats = df.groupby('user_id').agg(
        total_spent=('price', 'sum'),
        unique_categories=('category_code', 'nunique'),
        purchase_count=('price', 'count'),
        first_purchase=('event_time', 'min'),
        last_purchase=('event_time', 'max')
    ).reset_index()
    
    # Calculate days active
    user_stats['days_active'] = (user_stats['last_purchase'] - user_stats['first_purchase']).dt.days
    
    # Print diagnostic information
    print("\n=== Customer Purchase Statistics ===")
    print(f"Total unique users: {len(user_stats)}")
    print(f"Purchase count statistics:\n{user_stats['purchase_count'].describe()}")
    print(f"Total spent statistics:\n{user_stats['total_spent'].describe()}")
    
    # Gradually relaxing criteria
    criteria_sets = [
        {"total_spend": min_total_spend, "purchases": min_purchases},
        {"total_spend": 300, "purchases": 3},
        {"total_spend": 200, "purchases": 2},
        {"total_spend": 100, "purchases": 1}
    ]
    
    for criteria in criteria_sets:
        high_value_customers = user_stats[
            (user_stats['total_spent'] >= criteria['total_spend']) & 
            (user_stats['purchase_count'] >= criteria['purchases'])
        ].copy()
        
        print(f"\nCriteria: Total Spend >= ${criteria['total_spend']}, Purchases >= {criteria['purchases']}")
        print(f"High-value customers found: {len(high_value_customers)}")
        
        if len(high_value_customers) > 0:
            # Display top 10 high-value customers
            print("\nTop 10 High-Value Customers:")
            top_customers = high_value_customers.sort_values('total_spent', ascending=False).head(10)
            print(top_customers[['user_id', 'total_spent', 'purchase_count', 'unique_categories']])
            
            return high_value_customers
    
    print("No high-value customers found even with relaxed criteria.")
    return user_stats.head(20)  # Return top 20 users as a fallback

def forecast_product_demand(product_daily, product_id, forecast_days=30):
    """
    Forecast product demand with robust error handling.
    """
    print(f"Forecasting demand for product ID {product_id}...")
    
    # Filter data for the specific product
    product_data = product_daily[product_daily['product_id'] == product_id].copy()
    
    if len(product_data) < 10:
        print(f"Insufficient data for product ID {product_id}. Need at least 10 data points.")
        return None
    
    # Set date as index and sort
    product_data = product_data.set_index('date').sort_index()
    
    # Use units_sold for forecasting
    sales_series = product_data['units_sold']
    
    try:
        # Simplified ARIMA forecasting with error handling
        model = ARIMA(sales_series, order=(1,1,1))
        model_fit = model.fit()
        
        # Forecast future values
        forecast_result = model_fit.forecast(steps=forecast_days)
        forecast_index = pd.date_range(
            start=sales_series.index[-1] + pd.Timedelta(days=1), 
            periods=forecast_days
        )
        
        forecast_df = pd.DataFrame({
            'date': forecast_index,
            'forecast': forecast_result
        })
        
        # Clip negative values
        forecast_df['forecast'] = forecast_df['forecast'].clip(lower=0)
        
        print(f"Successfully generated {forecast_days}-day forecast for product {product_id}")
        
        return sales_series, forecast_df
    
    except Exception as e:
        print(f"Error forecasting product {product_id}: {str(e)}")
        return None

def corporate_purchase_prediction(high_value_customers, df, time_window=30):
    """
    Predict purchase behavior for high-value customers with improved time reference
    and cross-validation.
    
    Parameters:
    high_value_customers (DataFrame): DataFrame containing high-value customers
    df (DataFrame): Purchase data
    time_window (int): Number of days to use as a validation window
    
    Returns:
    DataFrame: Predictions with accuracy metrics
    """
    print("Predicting purchase behavior with cross-validation...")
    
    predictions = []
    validation_results = []
    
    # Use the max date in dataset minus time_window as reference point
    # This allows for validation against known purchases
    max_date = df['event_time'].max()
    validation_cutoff = max_date - pd.Timedelta(days=time_window)
    current_time = validation_cutoff
    
    print(f"Using {validation_cutoff} as reference point for predictions")
    print(f"Validation window: {validation_cutoff} to {max_date}")
    
    for _, customer in high_value_customers.iterrows():
        user_id = customer['user_id']
        user_purchases = df[df['user_id'] == user_id].sort_values('event_time')
        
        if len(user_purchases) < 2:
            continue
        
        # Ensure datetime is properly handled - don't try to add timezone if already has one
        # Check if timestamps already have timezone information
        if not pd.api.types.is_datetime64tz_dtype(user_purchases['event_time']):
            # Only localize if not already tz-aware
            user_purchases['event_time'] = pd.to_datetime(user_purchases['event_time']).dt.tz_localize('UTC')
        
        # Split into training and validation data
        training_purchases = user_purchases[user_purchases['event_time'] <= validation_cutoff]
        validation_purchases = user_purchases[user_purchases['event_time'] > validation_cutoff]
        
        if len(training_purchases) < 2:
            continue
            
        # Calculate purchase intervals using training data only
        training_purchases['next_purchase_time'] = training_purchases['event_time'].shift(-1)
        training_purchases['days_to_next_purchase'] = (
            training_purchases['next_purchase_time'] - training_purchases['event_time']
        ).dt.total_seconds() / (24 * 3600)
        
        # Calculate average days between purchases
        avg_days_between_purchases = training_purchases['days_to_next_purchase'].mean()
        if pd.isna(avg_days_between_purchases):
            avg_days_between_purchases = 30  # Default to 30 days if no pattern detected
        
        # Apply seasonality adjustment (weight recent intervals more)
        if len(training_purchases) >= 3:
            # Calculate weighted average with more recent purchases having higher weight
            weights = range(1, len(training_purchases))
            weighted_avg = np.average(
                training_purchases['days_to_next_purchase'].dropna(),
                weights=weights
            )
            avg_days_between_purchases = weighted_avg
        
        last_purchase_date = training_purchases['event_time'].max()
        predicted_next_purchase = last_purchase_date + pd.Timedelta(days=avg_days_between_purchases)
        
        # Validate if prediction was accurate (did customer purchase within ±3 days of prediction?)
        accuracy = 0
        purchase_happened = False
        days_off = None
        
        if len(validation_purchases) > 0:
            actual_next_purchase = validation_purchases['event_time'].min()
            purchase_happened = True
            days_off = (actual_next_purchase - predicted_next_purchase).days
            # Consider prediction accurate if within 3 days
            accuracy = 1 if abs(days_off) <= 3 else 0
        
        prediction = {
            'user_id': user_id,
            'total_spent': customer['total_spent'],
            'purchase_count': customer['purchase_count'],
            'avg_days_between_purchases': avg_days_between_purchases,
            'last_purchase_date': last_purchase_date,
            'predicted_next_purchase': predicted_next_purchase,
            'days_until_next_purchase': (predicted_next_purchase - current_time).days,
            'purchase_happened': purchase_happened,
            'actual_purchase_date': validation_purchases['event_time'].min() if purchase_happened else None,
            'prediction_accuracy': accuracy,
            'days_off': days_off
        }
        
        predictions.append(prediction)
        if purchase_happened:
            validation_results.append(accuracy)
    
    if not predictions:
        print("No purchase predictions could be generated.")
        return pd.DataFrame()
    
    predictions_df = pd.DataFrame(predictions)
    
    # Calculate overall model accuracy
    if validation_results:
        overall_accuracy = sum(validation_results) / len(validation_results) * 100
        print(f"\nOverall prediction accuracy: {overall_accuracy:.2f}%")
        print(f"Based on {len(validation_results)} validated predictions")
    
    return predictions_df.sort_values('days_until_next_purchase')

def advanced_purchase_prediction(high_value_customers, df, time_window=30):
    """
    More sophisticated purchase prediction using machine learning and time series
    analysis that accounts for seasonality and trends.
    
    Parameters:
    high_value_customers (DataFrame): DataFrame containing high-value customers
    df (DataFrame): Purchase data
    time_window (int): Number of days to use as a validation window
    
    Returns:
    DataFrame: Predictions with accuracy metrics
    """
    print("Performing advanced purchase behavior prediction...")
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    import statsmodels.api as sm
    
    # Use the max date in dataset minus time_window as reference point
    max_date = df['event_time'].max()
    validation_cutoff = max_date - pd.Timedelta(days=time_window)
    current_time = validation_cutoff
    
    predictions = []
    model_maes = []
    
    for _, customer in high_value_customers.iterrows():
        user_id = customer['user_id']
        user_purchases = df[df['user_id'] == user_id].sort_values('event_time')
        
        if len(user_purchases) < 5:  # Need more data points for these models
            continue
        
        # Split into training and validation
        training_purchases = user_purchases[user_purchases['event_time'] <= validation_cutoff]
        validation_purchases = user_purchases[user_purchases['event_time'] > validation_cutoff]
        
        if len(training_purchases) < 5 or len(validation_purchases) == 0:
            continue
        
        # Create features for ML approach
        user_features = []
        last_purchase_time = None
        
        for idx, purchase in training_purchases.iterrows():
            if last_purchase_time is not None:
                days_since_last = (purchase['event_time'] - last_purchase_time).days
                hour = purchase['event_time'].hour
                day_of_week = purchase['event_time'].dayofweek
                month = purchase['event_time'].month
                price = purchase['price']
                
                feature_row = {
                    'days_since_last': days_since_last,
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'is_weekend': 1 if day_of_week >= 5 else 0,
                    'month': month,
                    'price': price
                }
                user_features.append(feature_row)
            
            last_purchase_time = purchase['event_time']
        
        # Try different prediction models based on available data
        predictions_dict = {}
        
        # 1. Basic rolling average (baseline)
        training_purchases['next_purchase_time'] = training_purchases['event_time'].shift(-1)
        training_purchases['days_to_next'] = (
            training_purchases['next_purchase_time'] - training_purchases['event_time']
        ).dt.days
        avg_interval = training_purchases['days_to_next'].dropna().mean()
        predictions_dict['rolling_avg'] = {
            'days': avg_interval,
            'date': last_purchase_time + pd.Timedelta(days=avg_interval)
        }
        
        # 2. Try Random Forest if enough data points
        if len(user_features) >= 5:
            try:
                features_df = pd.DataFrame(user_features)
                X = features_df.drop('days_since_last', axis=1)
                y = features_df['days_since_last']
                
                model = RandomForestRegressor(n_estimators=50)
                model.fit(X, y)
                
                # Predict using last purchase features
                last_features = X.iloc[-1:].copy()
                rf_prediction = model.predict(last_features)[0]
                predictions_dict['random_forest'] = {
                    'days': rf_prediction,
                    'date': last_purchase_time + pd.Timedelta(days=rf_prediction)
                }
            except Exception as e:
                print(f"Error in RF model for user {user_id}: {e}")
        
        # 3. Try ARIMA for time series if enough sequential data
        if len(training_purchases) >= 7:
            try:
                # Create time series of purchase intervals
                interval_series = training_purchases['days_to_next'].dropna()
                
                # Fit ARIMA model
                model = sm.tsa.ARIMA(interval_series, order=(1,0,0))
                model_fit = model.fit()
                
                # Forecast next interval
                arima_forecast = model_fit.forecast(steps=1)[0]
                predictions_dict['arima'] = {
                    'days': arima_forecast,
                    'date': last_purchase_time + pd.Timedelta(days=arima_forecast)
                }
            except Exception as e:
                print(f"Error in ARIMA model for user {user_id}: {e}")
        
        # Evaluate which model is best using validation data
        actual_next_purchase = validation_purchases['event_time'].min()
        best_model = 'rolling_avg'  # Default
        best_error = float('inf')
        
        for model_name, prediction in predictions_dict.items():
            predicted_date = prediction['date']
            error_days = abs((actual_next_purchase - predicted_date).days)
            
            if error_days < best_error:
                best_error = error_days
                best_model = model_name
        
        # Use the best model for final prediction
        best_prediction = predictions_dict[best_model]
        prediction_accuracy = 1 if best_error <= 3 else 0  # Accurate if within 3 days
        
        prediction = {
            'user_id': user_id,
            'total_spent': customer['total_spent'],
            'last_purchase_date': last_purchase_time,
            'best_model': best_model,
            'available_models': list(predictions_dict.keys()),
            'predicted_days': best_prediction['days'],
            'predicted_next_purchase': best_prediction['date'],
            'days_until_next_purchase': (best_prediction['date'] - current_time).days,
            'actual_purchase_date': actual_next_purchase,
            'error_days': best_error,
            'prediction_accuracy': prediction_accuracy
        }
        
        predictions.append(prediction)
        model_maes.append(best_error)
    
    if not predictions:
        print("No advanced predictions could be generated.")
        return pd.DataFrame()
    
    predictions_df = pd.DataFrame(predictions)
    
    # Calculate overall model performance
    avg_mae = sum(model_maes) / len(model_maes)
    accuracy_pct = len([p for p in predictions if p['prediction_accuracy'] == 1]) / len(predictions) * 100
    
    print(f"\nAdvanced model results:")
    print(f"Average error: {avg_mae:.2f} days")
    print(f"Prediction accuracy (within 3 days): {accuracy_pct:.2f}%")
    print(f"Most frequently used model: {predictions_df['best_model'].value_counts().index[0]}")
    
    return predictions_df.sort_values('days_until_next_purchase')

def analyze_purchase_patterns_by_product(df):
    """
    Analyze purchase patterns at the product level to identify trends.
    
    Parameters:
    df (DataFrame): Prepared purchase data
    
    Returns:
    Tuple of DataFrames with product purchase patterns
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
    
    print(f"Analyzed purchase patterns for {len(product_details)} unique products")
    
    return product_daily, top_product_daily, product_details

def diagnose_ecommerce_data(df):
    """
    Provide comprehensive diagnostics of the e-commerce dataset.
    """
    print("\n=== E-commerce Dataset Diagnostic ===")
    
    # Basic dataset information
    print(f"Total records: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique products: {df['product_id'].nunique()}")
    
    # Event type distribution
    print("\nEvent Type Distribution:")
    print(df['event_type'].value_counts())
    
    # Temporal analysis
    print("\nTemporal Analysis:")
    df['event_date'] = df['event_time'].dt.date
    print("Date range:", df['event_date'].min(), "to", df['event_date'].max())
    
    # Price and category analysis
    print("\nPrice Statistics:")
    print(df['price'].describe())
    
    print("\nTop 10 Categories:")
    print(df['category_code'].value_counts().head(10))
    
    print("\nTop 10 Brands:")
    print(df['brand'].value_counts().head(10))


def main():
    file_path = 'data/ecommerce_behavior.csv'
    
    try:
        # Load and prepare data
        purchases_df = load_and_prepare_data(file_path)
        
        # Identify high-value customers
        high_value_customers = identify_high_value_customers(
            purchases_df, min_total_spend=500, min_purchases=3
        )
        
        # Check if any high-value customers were found
        if len(high_value_customers) == 0:
            print("No high-value customers found. Adjust criteria or check data.")
            return
        
        # Analyze purchase patterns by product
        product_daily, top_product_daily, product_details = analyze_purchase_patterns_by_product(purchases_df)
        
        # Forecast top product demands
        top_products = purchases_df['product_id'].value_counts().head(5).index.tolist()
        product_forecasts = {}
        
        for product_id in top_products:
            forecast_result = forecast_product_demand(product_daily, product_id)
            if forecast_result:
                product_forecasts[product_id] = forecast_result
        
        # Predict purchase behavior with improved methods
        purchase_predictions = corporate_purchase_prediction(high_value_customers, purchases_df, time_window=30)
        
        # Run advanced prediction if data is sufficient
        if len(high_value_customers) >= 5:
            print("\nRunning advanced prediction models...")
            advanced_predictions = advanced_purchase_prediction(high_value_customers, purchases_df, time_window=30)
            
            if not advanced_predictions.empty:
                print("\n=== Advanced Model Purchase Predictions ===")
                print(advanced_predictions[['user_id', 'best_model', 'days_until_next_purchase', 
                                          'error_days', 'prediction_accuracy']].head(10))
        
        # Print key results
        if not purchase_predictions.empty:
            print("\n=== High-Value Customer Purchase Predictions ===")
            print(purchase_predictions)
        else:
            print("No purchase predictions could be generated.")
        
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()