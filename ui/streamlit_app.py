import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import yaml
import torch
import sys
import logging
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingest.data_loader import load_csv
from src.profiling.data_profiler import DataProfiler
from src.missing_handler.missing_handler import MissingHandler
from src.anomaly_detector.anomaly_detector import AnomalyDetector
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.data_validation.validator import DataValidator
from src.ai_models.models.set_transformer import SetTransformerModel
# Import other models as needed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Voltage Network Analysis Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("⚡ Voltage Network Analysis Platform")
st.markdown("""
This platform provides comprehensive tools for analyzing and predicting voltage 
network behavior using advanced AI models.
""")

def load_config():
    """Load configuration files."""
    config_dir = Path(__file__).parent.parent / "config"
    
    with open(config_dir / "models.yaml", 'r') as f:
        models_config = yaml.safe_load(f)
    
    with open(config_dir / "preprocessing.yaml", 'r') as f:
        preprocessing_config = yaml.safe_load(f)
    
    return models_config, preprocessing_config

def plot_voltage_predictions(
    true_values: np.ndarray,
    predictions: np.ndarray,
    timestamps: pd.Series
):
    """Plot actual vs predicted voltage values."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=true_values,
        name="Actual",
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=predictions,
        name="Predicted",
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title="Voltage Predictions vs Actual Values",
        xaxis_title="Time",
        yaxis_title="Voltage (V)",
        hovermode='x unified'
    )
    
    return fig

def plot_metrics_history(metrics_history: list):
    """Plot training metrics history."""
    df = pd.DataFrame(metrics_history)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Loss", "MAE", "RMSE", "R²")
    )
    
    # Loss
    fig.add_trace(
        go.Scatter(y=df['loss'], name="Training Loss"),
        row=1, col=1
    )
    
    # MAE
    fig.add_trace(
        go.Scatter(y=df['mae'], name="MAE"),
        row=1, col=2
    )
    
    # RMSE
    fig.add_trace(
        go.Scatter(y=df['rmse'], name="RMSE"),
        row=2, col=1
    )
    
    # R²
    fig.add_trace(
        go.Scatter(y=df['r2'], name="R²"),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="Training Metrics")
    return fig

def main():
    # Load configurations
    models_config, preprocessing_config = load_config()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Data Upload & Profiling",
         "Data Preprocessing",
         "Model Training",
         "Predictions & Analysis"]
    )
    
    # Session state initialization
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if page == "Data Upload & Profiling":
        st.header("Data Upload & Profiling")
        
        uploaded_file = st.file_uploader(
            "Upload voltage network data (CSV)",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                # Load and display data
                df = load_csv(uploaded_file)
                st.session_state.data = df
                
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Data profiling
                if st.button("Generate Data Profile"):
                    profiler = DataProfiler()
                    with st.spinner("Generating profile report..."):
                        report_path = profiler.generate_report(
                            df,
                            "Voltage Network Data Profile"
                        )
                        st.success(f"Profile report generated: {report_path}")
                        
                        # Display basic statistics
                        st.subheader("Basic Statistics")
                        st.write(df.describe())
                        
                        # Plot voltage distribution
                        if 'voltage' in df.columns:
                            fig = px.histogram(
                                df,
                                x='voltage',
                                title='Voltage Distribution'
                            )
                            st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    elif page == "Data Preprocessing":
        st.header("Data Preprocessing")
        
        if st.session_state.data is None:
            st.warning("Please upload data first!")
            return
        
        df = st.session_state.data
        
        # Missing value handling
        st.subheader("Missing Value Handling")
        missing_handler = MissingHandler(preprocessing_config)
        
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            st.write("Columns with missing values:", missing_cols)
            
            for col in missing_cols:
                method = st.selectbox(
                    f"Select imputation method for {col}",
                    ['interpolate', 'forward_fill', 'backward_fill', 'model_impute']
                )
                
                if st.button(f"Handle missing values in {col}"):
                    with st.spinner(f"Processing {col}..."):
                        df = missing_handler.apply_config_strategy(df, col)
                        st.session_state.data = df
                        st.success(f"Handled missing values in {col}")
        
        # Anomaly detection
        st.subheader("Anomaly Detection")
        anomaly_detector = AnomalyDetector(preprocessing_config)
        
        if st.button("Detect Anomalies"):
            with st.spinner("Detecting anomalies..."):
                flags, details = anomaly_detector.detect_statistical(
                    df,
                    ['voltage', 'current', 'power']
                )
                
                st.write("Anomaly Detection Results:")
                st.write(details)
                
                if st.button("Remove Anomalies"):
                    df = anomaly_detector.remove_outliers(df, flags)
                    st.session_state.data = df
                    st.success("Anomalies removed")
        
        # Feature engineering
        st.subheader("Feature Engineering")
        engineer = FeatureEngineer(preprocessing_config)
        
        if st.button("Add Time Features"):
            with st.spinner("Adding time features..."):
                df = engineer.add_time_features(df)
                st.session_state.data = df
                st.success("Time features added")
        
        if st.button("Add Window Features"):
            window_size = st.slider("Window Size (hours)", 1, 24, 6)
            with st.spinner("Adding window features..."):
                df = engineer.window_agg(
                    df,
                    ['voltage', 'current', 'power'],
                    window_size
                )
                st.session_state.data = df
                st.success("Window features added")
    
    elif page == "Model Training":
        st.header("Model Training")
        
        if st.session_state.data is None:
            st.warning("Please upload and preprocess data first!")
            return
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            list(models_config.keys())
        )
        
        # Display model configuration
        st.subheader("Model Configuration")
        config = models_config[model_type]
        st.write(config)
        
        # Training parameters
        st.subheader("Training Parameters")
        epochs = st.slider("Number of epochs", 10, 100, config['epochs'])
        batch_size = st.slider("Batch size", 16, 128, config['batch_size'])
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Prepare data
                    from src.ai_models.train import make_datasets, train_model
                    
                    train_ds, val_ds, test_ds = make_datasets(
                        st.session_state.data,
                        window_size=24
                    )
                    
                    # Initialize model
                    if model_type == 'set_transformer':
                        model = SetTransformerModel(config)
                    # Add other model types here
                    
                    # Create data loaders
                    train_loader = torch.utils.data.DataLoader(
                        train_ds,
                        batch_size=batch_size,
                        shuffle=True
                    )
                    val_loader = torch.utils.data.DataLoader(
                        val_ds,
                        batch_size=batch_size
                    )
                    
                    # Train model
                    config['epochs'] = epochs
                    model = train_model(
                        model,
                        train_loader,
                        val_loader,
                        config
                    )
                    
                    st.session_state.model = model
                    st.success("Model training completed!")
                    
                    # Plot training history
                    st.subheader("Training History")
                    fig = plot_metrics_history(model.training_history)
                    st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
    elif page == "Predictions & Analysis":
        st.header("Predictions & Analysis")
        
        if st.session_state.model is None:
            st.warning("Please train a model first!")
            return
        
        # Make predictions
        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                try:
                    model = st.session_state.model
                    df = st.session_state.data
                    
                    # Prepare test data
                    from src.ai_models.train import make_datasets
                    _, _, test_ds = make_datasets(df, window_size=24)
                    test_loader = torch.utils.data.DataLoader(
                        test_ds,
                        batch_size=32
                    )
                    
                    # Generate predictions
                    model.eval()
                    predictions = []
                    true_values = []
                    
                    with torch.no_grad():
                        for x, y in test_loader:
                            pred = model(x)
                            predictions.extend(pred.cpu().numpy())
                            true_values.extend(y.cpu().numpy())
                    
                    # Plot results
                    timestamps = df.index[-len(predictions):]
                    fig = plot_voltage_predictions(
                        np.array(true_values),
                        np.array(predictions),
                        timestamps
                    )
                    st.plotly_chart(fig)
                    
                    # Display metrics
                    metrics = model.compute_metrics(
                        torch.tensor(predictions),
                        torch.tensor(true_values)
                    )
                    st.subheader("Prediction Metrics")
                    st.write(metrics)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}") 