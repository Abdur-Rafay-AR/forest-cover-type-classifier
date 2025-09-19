"""
Forest Cover Type Classification - Streamlit App
A machine learning web application for predicting forest cover types 
based on cartographic and environmental features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Forest Cover Type Classifier",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #2E8B57;
    }
</style>
""", unsafe_allow_html=True)

# Forest cover type mapping
COVER_TYPE_NAMES = {
    1: 'Spruce/Fir',
    2: 'Lodgepole Pine',
    3: 'Ponderosa Pine',
    4: 'Cottonwood/Willow',
    5: 'Aspen',
    6: 'Douglas-fir',
    7: 'Krummholz'
}

# Cover type descriptions
COVER_TYPE_DESCRIPTIONS = {
    1: "Dense coniferous forests, typically found at higher elevations",
    2: "Common pine species, adaptable to various conditions",
    3: "Large pine trees, prefer drier, warmer climates",
    4: "Deciduous trees near water sources",
    5: "Fast-growing deciduous trees, often in disturbed areas",
    6: "Large evergreen trees, commercially important",
    7: "Stunted trees at treeline, harsh mountain conditions"
}

@st.cache_resource
def load_models_and_data():
    """Load pre-trained models and sample data"""
    try:
        with open('models/rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/xgb_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('models/performance_metrics.pkl', 'rb') as f:
            performance_metrics = pickle.load(f)
        with open('models/sample_data.pkl', 'rb') as f:
            sample_data = pickle.load(f)
        return rf_model, xgb_model, feature_names, performance_metrics, sample_data
    except FileNotFoundError:
        st.error("Model files not found. Please run `python extract_models.py` first to train and save the models.")
        return None, None, None, None, None

def create_feature_input():
    """Create input widgets for all features"""
    st.sidebar.header("🌲 Forest Characteristics")
    
    # Quantitative features
    st.sidebar.subheader("📊 Environmental Features")
    
    elevation = st.sidebar.slider(
        "Elevation (meters)", 
        min_value=1800, max_value=3900, value=2800, step=10,
        help="Elevation in meters above sea level"
    )
    
    aspect = st.sidebar.slider(
        "Aspect (degrees)", 
        min_value=0, max_value=360, value=180, step=5,
        help="Compass bearing of slope aspect"
    )
    
    slope = st.sidebar.slider(
        "Slope (degrees)", 
        min_value=0, max_value=60, value=15, step=1,
        help="Slope steepness in degrees"
    )
    
    hydro_distance = st.sidebar.slider(
        "Distance to Water (meters)", 
        min_value=0, max_value=1400, value=300, step=10,
        help="Horizontal distance to nearest water feature"
    )
    
    hydro_vertical = st.sidebar.slider(
        "Vertical Distance to Water (meters)", 
        min_value=-200, max_value=600, value=50, step=5,
        help="Vertical distance to nearest water feature"
    )
    
    road_distance = st.sidebar.slider(
        "Distance to Roads (meters)", 
        min_value=0, max_value=7000, value=2000, step=50,
        help="Horizontal distance to nearest roadway"
    )
    
    fire_distance = st.sidebar.slider(
        "Distance to Fire Points (meters)", 
        min_value=0, max_value=7000, value=2000, step=50,
        help="Horizontal distance to nearest wildfire ignition points"
    )
    
    hillshade_9am = st.sidebar.slider(
        "Hillshade at 9am", 
        min_value=0, max_value=255, value=200, step=5,
        help="Amount of shade at 9am (0=full shade, 255=full sun)"
    )
    
    hillshade_noon = st.sidebar.slider(
        "Hillshade at Noon", 
        min_value=0, max_value=255, value=220, step=5,
        help="Amount of shade at noon"
    )
    
    hillshade_3pm = st.sidebar.slider(
        "Hillshade at 3pm", 
        min_value=0, max_value=255, value=150, step=5,
        help="Amount of shade at 3pm"
    )
    
    # Wilderness Area
    st.sidebar.subheader("🏔️ Wilderness Area")
    wilderness_area = st.sidebar.selectbox(
        "Select Wilderness Area",
        options=[1, 2, 3, 4],
        format_func=lambda x: f"Wilderness Area {x}",
        help="Designated wilderness area (1-4)"
    )
    
    # Soil Type
    st.sidebar.subheader("🌱 Soil Type")
    soil_type = st.sidebar.selectbox(
        "Select Soil Type",
        options=list(range(1, 41)),
        index=0,
        format_func=lambda x: f"Soil Type {x}",
        help="Soil type classification (1-40)"
    )
    
    return {
        'Elevation': elevation,
        'Aspect': aspect,
        'Slope': slope,
        'Horizontal_Distance_To_Hydrology': hydro_distance,
        'Vertical_Distance_To_Hydrology': hydro_vertical,
        'Horizontal_Distance_To_Roadways': road_distance,
        'Horizontal_Distance_To_Fire_Points': fire_distance,
        'Hillshade_9am': hillshade_9am,
        'Hillshade_Noon': hillshade_noon,
        'Hillshade_3pm': hillshade_3pm,
        'wilderness_area': wilderness_area,
        'soil_type': soil_type
    }

def create_feature_vector(inputs, feature_names):
    """Convert user inputs to feature vector for prediction"""
    # Initialize feature vector
    features = np.zeros(len(feature_names))
    
    # Quantitative features
    quantitative_features = [
        'Elevation', 'Aspect', 'Slope', 
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 
        'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
    ]
    
    for i, feature in enumerate(quantitative_features):
        if feature in feature_names:
            idx = feature_names.index(feature)
            features[idx] = inputs[feature]
    
    # Wilderness areas (one-hot encoded)
    wilderness_col = f"Wilderness_Area_{inputs['wilderness_area']}"
    if wilderness_col in feature_names:
        idx = feature_names.index(wilderness_col)
        features[idx] = 1
    
    # Soil types (one-hot encoded)
    soil_col = f"Soil_Type_{inputs['soil_type']}"
    if soil_col in feature_names:
        idx = feature_names.index(soil_col)
        features[idx] = 1
    
    return features.reshape(1, -1)

def create_confusion_matrix_plot(sample_data, model_name):
    """Create confusion matrix visualization"""
    from sklearn.metrics import confusion_matrix
    
    if model_name == "Random Forest":
        y_true = sample_data['y_sample']
        y_pred = sample_data['rf_predictions']
    else:
        y_true = sample_data['y_sample']
        y_pred = sample_data['xgb_predictions']
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=list(COVER_TYPE_NAMES.values()),
        y=list(COVER_TYPE_NAMES.values()),
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f'{model_name} - Confusion Matrix (Sample Data)',
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        height=500
    )
    
    return fig

def create_feature_importance_plot(model, feature_names, model_name):
    """Create feature importance visualization"""
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)
    
    fig = px.bar(
        feature_imp_df, 
        x='importance', 
        y='feature',
        orientation='h',
        title=f"Top 15 Most Important Features ({model_name})",
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
    return fig

def create_model_comparison_chart(performance_metrics):
    """Create model performance comparison"""
    metrics = ['Training Accuracy', 'Test Accuracy']
    rf_scores = [performance_metrics['rf_train_accuracy'], performance_metrics['rf_test_accuracy']]
    xgb_scores = [performance_metrics['xgb_train_accuracy'], performance_metrics['xgb_test_accuracy']]
    
    fig = go.Figure(data=[
        go.Bar(name='Random Forest', x=metrics, y=rf_scores, marker_color='lightblue'),
        go.Bar(name='XGBoost', x=metrics, y=xgb_scores, marker_color='lightcoral')
    ])
    
    fig.update_layout(
        title='Model Performance Comparison',
        yaxis_title='Accuracy',
        barmode='group',
        height=400
    )
    
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">🌲 Forest Cover Type Classifier</h1>', unsafe_allow_html=True)
    st.markdown("Predict forest cover types using machine learning based on cartographic and environmental features")
    
    # Load models
    rf_model, xgb_model, feature_names, performance_metrics, sample_data = load_models_and_data()
    
    if rf_model is None:
        st.error("Please run `python extract_models.py` first to train and save the models!")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Model Performance", "🎯 Feature Importance", "📖 About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Make a Prediction</h2>', unsafe_allow_html=True)
        
        # Get user inputs
        inputs = create_feature_input()
        
        # Create feature vector
        feature_vector = create_feature_vector(inputs, feature_names)
        
        # Make predictions
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔮 Predict with Random Forest", type="primary"):
                rf_pred = rf_model.predict(feature_vector)[0]
                rf_proba = rf_model.predict_proba(feature_vector)[0]
                
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                st.markdown(f"### 🌲 Predicted Cover Type: {COVER_TYPE_NAMES[rf_pred]}")
                st.markdown(f"**Confidence:** {rf_proba[rf_pred-1]*100:.1f}%")
                st.markdown(f"*{COVER_TYPE_DESCRIPTIONS[rf_pred]}*")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show probability distribution
                fig = go.Figure(data=[
                    go.Bar(x=list(COVER_TYPE_NAMES.values()), y=rf_proba*100,
                           marker_color='lightblue')
                ])
                fig.update_layout(
                    title="Random Forest - Prediction Probabilities",
                    xaxis_title="Cover Type",
                    yaxis_title="Probability (%)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if st.button("⚡ Predict with XGBoost", type="primary"):
                xgb_pred = xgb_model.predict(feature_vector)[0] + 1  # Convert back to 1-7
                xgb_proba = xgb_model.predict_proba(feature_vector)[0]
                
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                st.markdown(f"### 🌲 Predicted Cover Type: {COVER_TYPE_NAMES[xgb_pred]}")
                st.markdown(f"**Confidence:** {xgb_proba[xgb_pred-1]*100:.1f}%")
                st.markdown(f"*{COVER_TYPE_DESCRIPTIONS[xgb_pred]}*")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show probability distribution
                fig = go.Figure(data=[
                    go.Bar(x=list(COVER_TYPE_NAMES.values()), y=xgb_proba*100,
                           marker_color='lightcoral')
                ])
                fig.update_layout(
                    title="XGBoost - Prediction Probabilities",
                    xaxis_title="Cover Type",
                    yaxis_title="Probability (%)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Display current input summary
        st.markdown("### 📋 Current Input Summary")
        input_df = pd.DataFrame([inputs]).T
        input_df.columns = ['Value']
        input_df.index.name = 'Feature'
        st.dataframe(input_df, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
        
        # Model performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("RF Training Accuracy", f"{performance_metrics['rf_train_accuracy']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("RF Test Accuracy", f"{performance_metrics['rf_test_accuracy']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("XGB Training Accuracy", f"{performance_metrics['xgb_train_accuracy']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            best_model = "Random Forest" if performance_metrics['rf_test_accuracy'] > performance_metrics['xgb_test_accuracy'] else "XGBoost"
            st.metric("Best Model", best_model)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance comparison chart
        fig = create_model_comparison_chart(performance_metrics)
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices
        st.markdown("### 🎯 Confusion Matrices (Sample Data)")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rf = create_confusion_matrix_plot(sample_data, "Random Forest")
            st.plotly_chart(fig_rf, use_container_width=True)
        
        with col2:
            fig_xgb = create_confusion_matrix_plot(sample_data, "XGBoost")
            st.plotly_chart(fig_xgb, use_container_width=True)
        
        # Model statistics
        st.markdown("### 📈 Detailed Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Training Accuracy', 'Test Accuracy', 'Overfitting Gap'],
            'Random Forest': [
                f"{performance_metrics['rf_train_accuracy']:.4f}",
                f"{performance_metrics['rf_test_accuracy']:.4f}",
                f"{performance_metrics['rf_train_accuracy'] - performance_metrics['rf_test_accuracy']:.4f}"
            ],
            'XGBoost': [
                f"{performance_metrics['xgb_train_accuracy']:.4f}",
                f"{performance_metrics['xgb_test_accuracy']:.4f}",
                f"{performance_metrics['xgb_train_accuracy'] - performance_metrics['xgb_test_accuracy']:.4f}"
            ]
        })
        st.dataframe(metrics_df, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Feature Importance Analysis</h2>', unsafe_allow_html=True)
        
        # Feature importance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rf = create_feature_importance_plot(rf_model, feature_names, "Random Forest")
            st.plotly_chart(fig_rf, use_container_width=True)
        
        with col2:
            fig_xgb = create_feature_importance_plot(xgb_model, feature_names, "XGBoost")
            st.plotly_chart(fig_xgb, use_container_width=True)
        
        # Feature importance comparison table
        st.markdown("### 📊 Feature Importance Comparison")
        rf_imp = pd.DataFrame({
            'Feature': feature_names,
            'RF_Importance': rf_model.feature_importances_
        }).sort_values('RF_Importance', ascending=False)
        
        xgb_imp = pd.DataFrame({
            'Feature': feature_names,
            'XGB_Importance': xgb_model.feature_importances_
        }).sort_values('XGB_Importance', ascending=False)
        
        # Merge and show top features
        comparison_df = pd.merge(rf_imp, xgb_imp, on='Feature')
        comparison_df['Avg_Importance'] = (comparison_df['RF_Importance'] + comparison_df['XGB_Importance']) / 2
        comparison_df = comparison_df.sort_values('Avg_Importance', ascending=False).head(20)
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Feature insights
        st.markdown("### 💡 Key Insights")
        st.info("""
        **Most Important Features:**
        - **Elevation**: Primary factor determining forest types at different altitudes
        - **Distance Features**: Proximity to water, roads, and fire points significantly impacts vegetation
        - **Soil Types**: Specific soil compositions strongly correlate with certain cover types
        - **Wilderness Areas**: Geographic regions have distinct ecological characteristics
        """)
    
    with tab4:
        st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)
        
        st.markdown(f"""
        ### 🌲 Forest Cover Type Classification
        
        This application uses machine learning to predict forest cover types based on cartographic variables. 
        The models were trained on the UCI Covertype dataset containing over 580,000 observations.
        
        **Dataset Features:**
        - 🏔️ **Elevation**: Height above sea level (1,800-3,900m)
        - 🧭 **Aspect**: Compass direction of slope (0-360°)
        - 📐 **Slope**: Steepness of terrain (0-60°)
        - 💧 **Water Distance**: Horizontal distance to nearest water features
        - ↕️ **Vertical Water Distance**: Vertical distance to water features
        - 🛣️ **Road Distance**: Distance to nearest roads
        - 🔥 **Fire Distance**: Distance to fire ignition points
        - ☀️ **Hillshade**: Amount of sunlight at 9am, noon, and 3pm
        - 🏔️ **Wilderness Areas**: 4 designated wilderness areas
        - 🌱 **Soil Types**: 40 different soil classifications
        
        **Forest Cover Types:**
        1. **Spruce/Fir**: Dense coniferous forests at higher elevations
        2. **Lodgepole Pine**: Adaptable pine species
        3. **Ponderosa Pine**: Large pines in drier climates
        4. **Cottonwood/Willow**: Deciduous trees near water
        5. **Aspen**: Fast-growing deciduous trees
        6. **Douglas-fir**: Large commercial evergreens
        7. **Krummholz**: Stunted trees at treeline
        
        **Models Used:**
        - 🌳 **Random Forest**: Ensemble of decision trees
        - ⚡ **XGBoost**: Gradient boosting algorithm
        
        **Performance:**
        - Random Forest achieved {performance_metrics['rf_test_accuracy']:.1%} test accuracy
        - XGBoost achieved {performance_metrics['xgb_test_accuracy']:.1%} test accuracy
        
        **Usage Instructions:**
        1. Use the sidebar to input forest characteristics
        2. Click on either model to get predictions
        3. Explore model performance and feature importance in other tabs
        4. Use the predictions for forest management and planning
        
        **Data Source:** UCI Machine Learning Repository - Covertype Data Set
        """)
        
        # Add sample predictions showcase
        st.markdown("### 🔍 Sample Predictions")
        if st.button("Show Random Sample Predictions"):
            sample_indices = np.random.choice(len(sample_data['X_sample']), 5)
            
            for i, idx in enumerate(sample_indices):
                with st.expander(f"Sample {i+1}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Input Features:**")
                        sample_features = sample_data['X_sample'].iloc[idx]
                        # Show only key features for brevity
                        key_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology']
                        for feature in key_features:
                            if feature in sample_features.index:
                                st.write(f"- {feature}: {sample_features[feature]:.0f}")
                    
                    with col2:
                        st.write("**Predictions:**")
                        true_label = sample_data['y_sample'].iloc[idx]
                        rf_pred = sample_data['rf_predictions'][idx]
                        xgb_pred = sample_data['xgb_predictions'][idx]
                        
                        st.write(f"- **True**: {COVER_TYPE_NAMES[true_label]}")
                        st.write(f"- **RF**: {COVER_TYPE_NAMES[rf_pred]} {'✅' if rf_pred == true_label else '❌'}")
                        st.write(f"- **XGB**: {COVER_TYPE_NAMES[xgb_pred]} {'✅' if xgb_pred == true_label else '❌'}")
                    
                    with col3:
                        st.write("**Confidence:**")
                        rf_conf = sample_data['rf_probabilities'][idx][rf_pred-1] * 100
                        xgb_conf = sample_data['xgb_probabilities'][idx][xgb_pred-1] * 100
                        
                        st.write(f"- **RF**: {rf_conf:.1f}%")
                        st.write(f"- **XGB**: {xgb_conf:.1f}%")

if __name__ == "__main__":
    main()