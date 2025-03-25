# E-commerce Behavior Analysis Project

## Dataset Source
The dataset used in this project is from Kaggle: [E-commerce Behavior Data from Multi-Category Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store/data)

## Project Overview
This project provides a comprehensive analysis of e-commerce behavior using advanced machine learning techniques, including predictive modeling, customer segmentation, and purchase pattern analysis.

## Project Files

### 1. `main.py` - Project Orchestration and Workflow Management
- Serves as the central entry point for the entire project
- Manages the overall data processing and analysis workflow
- Key responsibilities:
  - Data sampling from the original large dataset
  - Coordinating data analysis steps
  - Initiating model training
  - Handling file paths and data dependencies
- Handles error checking and provides informative logging
- Ensures reproducibility of the analysis process

### 2. `revised_purchase_prediction.py` - Comprehensive E-commerce Behavior Analysis
- Advanced predictive analytics for understanding customer purchase behavior
- Key functions and capabilities:
  - Data loading and preprocessing
  - High-value customer identification
  - Purchase pattern analysis
  - Product demand forecasting
  - Advanced purchase prediction using multiple models
- Sophisticated prediction techniques include:
  - Rolling average prediction
  - Random Forest regression
  - ARIMA time series forecasting
- Provides detailed insights into:
  - Customer purchase intervals
  - Prediction accuracy
  - Model performance comparisons
- Generates comprehensive diagnostic information about e-commerce data
- Supports flexible criteria for customer segmentation
- Implements cross-validation for robust predictions

### 3. `ann.py` - Artificial Neural Network Model
- Implements a deep learning approach for predicting purchase events
- Uses TensorFlow/Keras to create a neural network classifier
- Preprocessing of e-commerce data
- Model training and evaluation
- Visualization of model performance (training history, confusion matrix)

### 4. `data_analysis.py` - Exploratory Data Analysis
- Performs in-depth analysis of the e-commerce dataset
- Generates visualizations of:
  - Event type distributions
  - Price distributions
  - Events by hour and day of week
  - Top product categories and brands
- Creates summary reports of key business insights

### 5. `test_prediction.py` - Advanced Predictive Modeling
- Implements advanced customer behavior prediction techniques
- Key features:
  - High-value customer identification
  - Customer clustering using KMeans
  - Time series forecasting with ARIMA models
  - Parallel processing for scalable predictions
  - Prediction accuracy validation
  - Visualization of prediction performance

## Project Dependencies

### Python Version
- Python 3.10.0 (recommended)

### Core Data Science Packages
- numpy==1.24.3
- pandas==2.0.3
- matplotlib==3.7.2
- seaborn==0.12.2

### Machine Learning Packages
- scikit-learn==1.3.0
- tensorflow==2.13.0
- keras==2.13.1
- xgboost==1.7.6
- statsmodels==0.14.0

### Utilities
- tqdm==4.65.0
- ipykernel==6.25.0
- jupyter==1.0.0

### File Handling
- openpyxl==3.1.2

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Prerequisites
- Recommended Python 3.8+
- Pip package manager
- Virtual environment (optional but recommended)

## Usage
1. Download the dataset from the Kaggle link
2. Place the CSV files in a `data/` directory
3. Run the scripts individually or use the `main.py` script as the primary entry point

## Key Analyses
- Purchase prediction
- Corporate customer segmentation
- Behavioral pattern identification
- Time series forecasting of customer purchases
- Product demand prediction
- Seasonal trend analysis

## Output
The project generates:
- Detailed prediction reports
- Comprehensive visualizations
- Performance metrics
- Customer segmentation insights
- Product demand forecasts

## Workflow
1. Data preprocessing and sampling
2. Exploratory data analysis
3. Corporate customer identification
4. Customer clustering
5. Purchase pattern analysis
6. Predictive modeling
7. Visualization and reporting

## Note
- Ensure you have the required dependencies installed
- Use a compatible Python environment
- The project is designed to handle large e-commerce datasets
- Customize parameters in each script as needed for your specific use case
