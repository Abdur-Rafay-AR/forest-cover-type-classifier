# 🌲 Forest Cover Type Classifier

A machine learning web application for predicting forest cover types based on cartographic and environmental features using the UCI Forest CoverType dataset.

![Forest Cover Types](https://img.shields.io/badge/Forest%20Types-7%20Classes-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20XGBoost-orange)

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Forest Cover Types](#forest-cover-types)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This project implements a machine learning solution to predict forest cover types in the Roosevelt National Forest of northern Colorado. The application uses various cartographic and environmental features to classify areas into one of seven forest cover types.

The project includes:
- **Data Analysis**: Comprehensive exploratory data analysis in Jupyter notebook
- **Machine Learning Models**: Random Forest and XGBoost classifiers
- **Web Application**: Interactive Streamlit app for real-time predictions
- **Model Persistence**: Trained models saved for production use

## 📊 Dataset

The [Forest CoverType dataset](https://archive.ics.uci.edu/ml/datasets/covertype) from the UCI Machine Learning Repository contains:

- **581,012 observations** with **54 features**
- **12 cartographic measures** (elevation, slope, aspect, distances to water/roads/fire points)
- **4 wilderness areas** (binary indicators)
- **40 soil types** (binary indicators)
- **7 forest cover types** (target classes)

### Data Source
- **Original Owners**: Remote Sensing and GIS Program, Department of Forest Sciences, Colorado State University
- **Study Area**: Roosevelt National Forest, northern Colorado
- **Collection Method**: 30m x 30m cartographic data

## 🌳 Forest Cover Types

| Type | Name | Description |
|------|------|-------------|
| 1 | **Spruce/Fir** | Dense coniferous forests, typically found at higher elevations |
| 2 | **Lodgepole Pine** | Common pine species, adaptable to various conditions |
| 3 | **Ponderosa Pine** | Large pine trees, prefer drier, warmer climates |
| 4 | **Cottonwood/Willow** | Deciduous trees near water sources |
| 5 | **Aspen** | Fast-growing deciduous trees, often in disturbed areas |
| 6 | **Douglas-fir** | Large evergreen trees, commercially important |
| 7 | **Krummholz** | Stunted trees at treeline, harsh mountain conditions |

## ✨ Features

### 🔍 Interactive Web Application
- **Real-time Predictions**: Input environmental parameters and get instant forest type predictions
- **Model Comparison**: Compare predictions from Random Forest and XGBoost models
- **Confidence Scores**: View prediction probabilities for all forest types
- **Feature Visualization**: Interactive charts and plots for data exploration
- **Model Performance**: View accuracy metrics and confusion matrices

### 📈 Machine Learning Models
- **Random Forest Classifier**: Ensemble method with excellent interpretability
- **XGBoost Classifier**: Gradient boosting for high-performance predictions
- **Feature Importance**: Analysis of which environmental factors matter most
- **Cross-validation**: Robust model evaluation and hyperparameter tuning

### 📊 Data Analysis
- **Exploratory Data Analysis**: Comprehensive data visualization and statistics
- **Feature Engineering**: Correlation analysis and feature selection
- **Data Preprocessing**: Handling categorical variables and scaling

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Abdur-Rafay-AR/forest-cover-type-classifier.git
   cd forest-cover-type-classifier
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Extract and train models**
   ```bash
   python extract_models.py
   ```
   This will:
   - Load and preprocess the dataset
   - Train Random Forest and XGBoost models
   - Save trained models to the `models/` directory
   - Generate performance metrics

## 🎯 Usage

### Running the Web Application

```bash
streamlit run forest_cover_app.py
```

The application will open in your browser at `http://localhost:8501`.

### Using the Application

1. **Input Features**: Use the sidebar to input environmental parameters:
   - Elevation, slope, aspect
   - Distances to hydrological and road features
   - Wilderness area designation
   - Soil type

2. **Get Predictions**: 
   - Choose between Random Forest or XGBoost models
   - View predicted forest cover type
   - See confidence scores for all types

3. **Explore Data**:
   - View sample data and statistics
   - Examine feature distributions
   - Analyze model performance metrics

### Jupyter Notebook Analysis

Open the comprehensive analysis notebook:
```bash
jupyter notebook forest_cover_classification.ipynb
```

## 📈 Model Performance

### Random Forest Classifier
- **Accuracy**: ~96%
- **Features**: 54 input features
- **Interpretability**: High (feature importance available)

### XGBoost Classifier  
- **Accuracy**: ~97%
- **Features**: 54 input features
- **Performance**: Optimized for speed and accuracy

*Note: Exact performance metrics are generated during model training and stored in `models/performance_metrics.pkl`*

## 📁 Project Structure

```
forest-cover-type-classifier/
├── 📊 covtype.data.gz              # Compressed dataset
├── 📄 covtype.info                 # Dataset documentation
├── 📓 forest_cover_classification.ipynb  # Main analysis notebook
├── 🐍 extract_models.py           # Model training script
├── 🌐 forest_cover_app.py         # Streamlit web application
├── 📁 models/                     # Trained models and data
│   ├── rf_model.pkl              # Random Forest model
│   ├── xgb_model.pkl             # XGBoost model
│   ├── feature_names.pkl         # Feature column names
│   ├── performance_metrics.pkl   # Model evaluation metrics
│   └── sample_data.pkl           # Sample data for app
├── 🚫 .gitignore                 # Git ignore file
└── 📖 README.md                  # This file
```

## 🛠️ Technical Details

### Dependencies
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **xgboost**: Gradient boosting framework
- **matplotlib/seaborn/plotly**: Data visualization
- **pickle**: Model serialization

### Model Training Process
1. **Data Loading**: Load compressed dataset from `covtype.data.gz`
2. **Data Preprocessing**: Handle categorical variables and feature scaling
3. **Model Training**: Train Random Forest and XGBoost classifiers
4. **Model Evaluation**: Cross-validation and performance metrics
5. **Model Persistence**: Save models and metadata using pickle

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Forest CoverType dataset
- Colorado State University for the original data collection
- Jock A. Blackard and Dr. Denis J. Dean for dataset creation

## 📞 Contact

**Abdur Rafay**
- GitHub: [@Abdur-Rafay-AR](https://github.com/Abdur-Rafay-AR)
- Project Link: [https://github.com/Abdur-Rafay-AR/forest-cover-type-classifier](https://github.com/Abdur-Rafay-AR/forest-cover-type-classifier)

---

⭐ **Star this repository if you found it helpful!**