import os
import pandas as pd
from data_analysis import load_and_analyze_data, plot_data_distributions, plot_category_analysis, analyze_purchase_patterns, generate_summary_report
from ann import load_and_preprocess_data, train_ann_model, train_random_forest_model

def prepare_data_sample(file_path, sample_size=100000, output_path='data/ecommerce_behavior.csv'):
    """
    Load a sample of the Kaggle dataset to make it more manageable.
    Useful for initial testing as the original dataset is very large.
    
    Parameters:
    file_path (str): Path to the original dataset
    sample_size (int): Number of rows to sample
    output_path (str): Path to save the sampled dataset
    """
    try:
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        print(f"Original dataset shape: {df.shape}")
        
        # Take a stratified sample by event_type to maintain the distribution
        df_sampled = df.groupby('event_type', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(sample_size * len(x) / len(df))))
        )
        
        print(f"Sampled dataset shape: {df_sampled.shape}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the sampled dataset
        df_sampled.to_csv(output_path, index=False)
        print(f"Sampled dataset saved to {output_path}")
        
    except Exception as e:
        print(f"Error sampling data: {e}")
        raise

def main():
    # Define paths
    kaggle_path = 'data/2019-Nov.csv'  # Original Kaggle dataset
    sample_path = 'data/ecommerce_behavior.csv'  # Sampled dataset
    
    try: 
        # Check if the sample dataset already exists
        if not os.path.exists(sample_path):
            # Check if the original Kaggle dataset exists
            if os.path.exists(kaggle_path):
                print("Creating a manageable sample from the original dataset...")
                prepare_data_sample(kaggle_path)
            else:
                print(f"Error: Could not find the original dataset at {kaggle_path}")
                print("Please download the dataset from Kaggle and place it in the data directory.")
                print("URL: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store")
                return
        
        # Step 1: Perform data analysis
        print("\nStarting data analysis...")
        df = load_and_analyze_data(sample_path)
        plot_data_distributions(df)
        plot_category_analysis(df)
        analyze_purchase_patterns(df)
        generate_summary_report(df)
        
        # Step 2: Train and evaluate models
        print("\nStarting model training...")
        X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(sample_path)
        
        # Train ANN model
        ann_model = train_ann_model(X_train, X_test, y_train, y_test)
        
        # Train Random Forest model
        rf_model = train_random_forest_model(X_train, X_test, y_train, y_test, feature_names)
        
        print("\nProject execution completed successfully.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the file path is correct and the file exists.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == '__main__':
    main()