"""
Model Extraction Script
Extract trained models from the Jupyter notebook and save them for the Streamlit app
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
import gzip
import warnings
warnings.filterwarnings('ignore')

def create_models_directory():
    """Create models directory if it doesn't exist"""
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models' directory")

def load_and_prepare_data():
    """Load the dataset and prepare features"""
    print("Loading dataset...")
    
    # Load the dataset (same as in notebook)
    with gzip.open('covtype.data.gz', 'rt') as f:
        data = pd.read_csv(f, header=None)
    
    # Define column names
    quantitative_features = [
        'Elevation', 'Aspect', 'Slope', 
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 
        'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
    ]
    
    wilderness_areas = [f'Wilderness_Area_{i}' for i in range(1, 5)]
    soil_types = [f'Soil_Type_{i}' for i in range(1, 41)]
    target = ['Cover_Type']
    
    column_names = quantitative_features + wilderness_areas + soil_types + target
    data.columns = column_names
    
    print(f"Dataset loaded: {data.shape}")
    return data, column_names[:-1]  # Exclude target from feature names

def train_and_save_models():
    """Train models and save them as pickle files"""
    
    # Create models directory
    create_models_directory()
    
    # Load and prepare data
    data, feature_names = load_and_prepare_data()
    
    # Prepare features and target
    X = data.drop('Cover_Type', axis=1)
    y = data['Cover_Type']
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Calculate RF accuracy
    rf_train_acc = rf_model.score(X_train, y_train)
    rf_test_acc = rf_model.score(X_test, y_test)
    print(f"Random Forest - Train Acc: {rf_train_acc:.4f}, Test Acc: {rf_test_acc:.4f}")
    
    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        verbosity=0
    )
    
    # XGBoost needs 0-indexed labels
    y_train_xgb = y_train - 1
    y_test_xgb = y_test - 1
    xgb_model.fit(X_train, y_train_xgb)
    
    # Calculate XGB accuracy
    xgb_train_acc = xgb_model.score(X_train, y_train_xgb)
    xgb_test_acc = xgb_model.score(X_test, y_test_xgb)
    print(f"XGBoost - Train Acc: {xgb_train_acc:.4f}, Test Acc: {xgb_test_acc:.4f}")
    
    # Save models
    print("Saving models...")
    
    # Save Random Forest
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("✓ Random Forest model saved")
    
    # Save XGBoost
    with open('models/xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print("✓ XGBoost model saved")
    
    # Save feature names
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("✓ Feature names saved")
    
    # Save model performance metrics
    performance_metrics = {
        'rf_train_accuracy': rf_train_acc,
        'rf_test_accuracy': rf_test_acc,
        'xgb_train_accuracy': xgb_train_acc,
        'xgb_test_accuracy': xgb_test_acc,
        'feature_names': feature_names,
        'cover_type_names': {
            1: 'Spruce/Fir', 2: 'Lodgepole Pine', 3: 'Ponderosa Pine',
            4: 'Cottonwood/Willow', 5: 'Aspen', 6: 'Douglas-fir', 7: 'Krummholz'
        }
    }
    
    with open('models/performance_metrics.pkl', 'wb') as f:
        pickle.dump(performance_metrics, f)
    print("✓ Performance metrics saved")
    
    # Save sample predictions for testing
    sample_indices = np.random.choice(len(X_test), 100, replace=False)
    X_sample = X_test.iloc[sample_indices]
    y_sample = y_test.iloc[sample_indices]
    
    sample_data = {
        'X_sample': X_sample,
        'y_sample': y_sample,
        'rf_predictions': rf_model.predict(X_sample),
        'xgb_predictions': xgb_model.predict(X_sample) + 1,  # Convert back to 1-7
        'rf_probabilities': rf_model.predict_proba(X_sample),
        'xgb_probabilities': xgb_model.predict_proba(X_sample)
    }
    
    with open('models/sample_data.pkl', 'wb') as f:
        pickle.dump(sample_data, f)
    print("✓ Sample data saved")
    
    print("\n🎉 All models and data saved successfully!")
    print("Files created:")
    print("  - models/rf_model.pkl")
    print("  - models/xgb_model.pkl") 
    print("  - models/feature_names.pkl")
    print("  - models/performance_metrics.pkl")
    print("  - models/sample_data.pkl")
    
    return rf_model, xgb_model, feature_names, performance_metrics

if __name__ == "__main__":
    print("🌲 Forest Cover Type Classification - Model Extraction")
    print("=" * 60)
    
    try:
        train_and_save_models()
        print("\n✅ Model extraction completed successfully!")
        print("You can now run the Streamlit app with: streamlit run forest_cover_app.py")
    except Exception as e:
        print(f"\n❌ Error during model extraction: {str(e)}")
        raise