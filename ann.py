import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def load_and_preprocess_data(file_path, target='purchase'):
    """
    Load and preprocess the e-commerce behavior data.
    
    Parameters:
    file_path (str): Path to the dataset
    target (str): Target variable, default is 'purchase' which creates a binary classification 
                 problem (purchase vs. non-purchase events)
    
    Returns:
    X_train, X_test, y_train, y_test, feature_names
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the dataset at {file_path}")
    
    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        
        # Convert event_time to datetime
        data['event_time'] = pd.to_datetime(data['event_time'])
        
        # Feature engineering
        # Extract time-based features
        data['hour'] = data['event_time'].dt.hour
        data['day_of_week'] = data['event_time'].dt.dayofweek
        data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Handle missing values
        data['category_code'] = data['category_code'].fillna('unknown')
        data['brand'] = data['brand'].fillna('unknown')
        
        # Create the target variable - Binary classification problem
        if target == 'purchase':
            data['is_purchase'] = (data['event_type'] == 'purchase').astype(int)
            target_col = 'is_purchase'
        else:
            target_col = target
        
        # Select relevant features
        features = [
            'hour', 'day_of_week', 'is_weekend', 'price',
            'category_id', 'category_code', 'brand'
        ]
        
        # For this example, we'll limit the number of categories to prevent memory issues
        # Keep only the top N most frequent values for high-cardinality categorical features
        top_n = 100
        
        for col in ['category_code', 'brand']:
            top_values = data[col].value_counts().nlargest(top_n).index
            data.loc[~data[col].isin(top_values), col] = 'other'
        
        # Split features and target
        X = data[features]
        y = data[target_col]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define categorical and numerical features
        categorical_features = ['category_code', 'brand']
        numerical_features = ['hour', 'day_of_week', 'is_weekend', 'price', 'category_id']
        
        # Create preprocessing pipelines
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Fit and transform the data
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Get feature names after one-hot encoding
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        categorical_feature_names = ohe.get_feature_names_out(categorical_features)
        feature_names = numerical_features + list(categorical_feature_names)
        
        return X_train_transformed, X_test_transformed, y_train, y_test, feature_names
    
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        raise

def create_ann_model(input_shape):
    """Create an Artificial Neural Network model."""
    try:
        model = Sequential([
            # Input layer
            Dense(128, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    except Exception as e:
        print(f"Error creating ANN model: {e}")
        raise

def train_ann_model(X_train, X_test, y_train, y_test):
    """Train the ANN model and evaluate its performance."""
    try:
        # Create ANN model
        model = create_ann_model(X_train.shape[1])
        
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=256,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate the model
        print("\nANN Model Evaluation:")
        test_results = model.evaluate(X_test, y_test, verbose=0)
        metrics = ['Loss', 'Accuracy', 'AUC', 'Precision', 'Recall']
        for metric, value in zip(metrics, test_results):
            print(f'{metric}: {value:.4f}')
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred_binary, title="ANN Model - Confusion Matrix")
        
        # Print classification report
        print('\nANN Classification Report:')
        print(classification_report(y_test, y_pred_binary))
        
        return model
    
    except Exception as e:
        print(f"An error occurred during ANN model training: {e}")
        raise

def train_random_forest_model(X_train, X_test, y_train, y_test, feature_names=None):
    """Train a Random Forest model and evaluate its performance."""
    try:
        print("\nTraining Random Forest Model...")
        
        # Create and train Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nRandom Forest Model Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, title="Random Forest - Confusion Matrix")
        
        # Print classification report
        print('\nRandom Forest Classification Report:')
        print(classification_report(y_test, y_pred))
        
        # Plot feature importance if feature names are provided
        if feature_names is not None:
            plot_feature_importance(rf_model, feature_names)
        
        return rf_model
    
    except Exception as e:
        print(f"An error occurred during Random Forest model training: {e}")
        raise

def plot_training_history(history):
    """Plot the training history of the neural network."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(['Train', 'Validation'])
        
        # Plot loss
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(['Train', 'Validation'])
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting training history: {e}")

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot a confusion matrix."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def plot_feature_importance(model, feature_names):
    """Plot feature importances for the Random Forest model."""
    try:
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Only show top 20 features if there are many
        if len(feature_names) > 20:
            indices = indices[:20]
            title = 'Top 20 Feature Importances'
        else:
            title = 'Feature Importances'
        
        # Plot the feature importances
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, len(indices)])
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting feature importance: {e}")

def main():
    # Define the file path
    file_path = 'data/ecommerce_behavior.csv'
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(file_path)
        
        # Train and evaluate models
        ann_model = train_ann_model(X_train, X_test, y_train, y_test)
        rf_model = train_random_forest_model(X_train, X_test, y_train, y_test, feature_names)
        
        print("\nModel training completed successfully.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the file 'ecommerce_behavior.csv' is in the data directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()