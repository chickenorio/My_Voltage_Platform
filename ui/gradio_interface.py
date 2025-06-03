import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
from src.ai_models.train import make_datasets, train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration files."""
    config_dir = Path(__file__).parent.parent / "config"
    
    with open(config_dir / "models.yaml", 'r') as f:
        models_config = yaml.safe_load(f)
    
    with open(config_dir / "preprocessing.yaml", 'r') as f:
        preprocessing_config = yaml.safe_load(f)
    
    return models_config, preprocessing_config

def process_data(
    file: str,
    handle_missing: bool = True,
    detect_anomalies: bool = True,
    add_features: bool = True
) -> tuple:
    """
    Process uploaded data file.
    
    Args:
        file: Path to uploaded file
        handle_missing: Whether to handle missing values
        detect_anomalies: Whether to detect anomalies
        add_features: Whether to add engineered features
        
    Returns:
        Tuple of (processed DataFrame, processing report)
    """
    try:
        # Load configurations
        _, preprocessing_config = load_config()
        
        # Load data
        df = load_csv(file)
        report = ["Data loaded successfully"]
        
        if handle_missing:
            # Handle missing values
            missing_handler = MissingHandler(preprocessing_config)
            missing_cols = df.columns[df.isnull().any()].tolist()
            
            if missing_cols:
                for col in missing_cols:
                    df = missing_handler.apply_config_strategy(df, col)
                report.append(f"Handled missing values in {len(missing_cols)} columns")
        
        if detect_anomalies:
            # Detect and remove anomalies
            anomaly_detector = AnomalyDetector(preprocessing_config)
            flags, details = anomaly_detector.detect_statistical(
                df,
                ['voltage', 'current', 'power']
            )
            df = anomaly_detector.remove_outliers(df, flags)
            report.append(
                f"Removed {len(df[flags.any(axis=1)])} anomalous records"
            )
        
        if add_features:
            # Add engineered features
            engineer = FeatureEngineer(preprocessing_config)
            df = engineer.add_time_features(df)
            df = engineer.window_agg(
                df,
                ['voltage', 'current', 'power'],
                window_sizes=[6, 12, 24]
            )
            report.append("Added time and window features")
        
        return df, "\n".join(report)
    
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        return None, f"Error: {str(e)}"

def train_and_evaluate(
    df: pd.DataFrame,
    model_type: str,
    epochs: int,
    batch_size: int
) -> tuple:
    """
    Train model and evaluate performance.
    
    Args:
        df: Input DataFrame
        model_type: Type of model to train
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Tuple of (trained model, training report, performance plot)
    """
    try:
        # Load model configuration
        models_config, _ = load_config()
        config = models_config[model_type]
        config['epochs'] = epochs
        config['batch_size'] = batch_size
        
        # Prepare datasets
        train_ds, val_ds, test_ds = make_datasets(df, window_size=24)
        
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size
        )
        
        # Initialize and train model
        if model_type == 'set_transformer':
            model = SetTransformerModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = train_model(
            model,
            train_loader,
            val_loader,
            config
        )
        
        # Generate performance plot
        history = pd.DataFrame(model.training_history)
        fig = go.Figure()
        
        # Training loss
        fig.add_trace(go.Scatter(
            y=history['loss'],
            name='Training Loss',
            line=dict(color='blue')
        ))
        
        # Validation loss
        if 'val_loss' in history.columns:
            fig.add_trace(go.Scatter(
                y=history['val_loss'],
                name='Validation Loss',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title='Training History',
            xaxis_title='Epoch',
            yaxis_title='Loss'
        )
        
        # Generate report
        final_metrics = history.iloc[-1].to_dict()
        report = [
            f"Training completed in {len(history)} epochs",
            f"Final loss: {final_metrics['loss']:.4f}",
            f"Final MAE: {final_metrics.get('mae', 'N/A')}"
        ]
        
        return model, "\n".join(report), fig
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return None, f"Error: {str(e)}", None

def make_predictions(
    model: torch.nn.Module,
    df: pd.DataFrame,
    window_size: int = 24
) -> tuple:
    """
    Generate predictions using trained model.
    
    Args:
        model: Trained model
        df: Input DataFrame
        window_size: Size of input window
        
    Returns:
        Tuple of (predictions plot, metrics report)
    """
    try:
        # Prepare test data
        _, _, test_ds = make_datasets(df, window_size=window_size)
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
        
        # Create plot
        timestamps = df.index[-len(predictions):]
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=true_values,
            name='Actual',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=predictions,
            name='Predicted',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Voltage Predictions vs Actual Values',
            xaxis_title='Time',
            yaxis_title='Voltage (V)'
        )
        
        # Calculate metrics
        metrics = model.compute_metrics(
            torch.tensor(predictions),
            torch.tensor(true_values)
        )
        
        report = [
            "Prediction Metrics:",
            f"MAE: {metrics['mae']:.4f}",
            f"RMSE: {metrics['rmse']:.4f}",
            f"R²: {metrics['r2']:.4f}"
        ]
        
        return fig, "\n".join(report)
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return None, f"Error: {str(e)}"

def create_interface():
    """Create Gradio interface."""
    
    # Load configurations
    models_config, _ = load_config()
    
    with gr.Blocks(title="Voltage Network Analysis Platform") as interface:
        gr.Markdown("# ⚡ Voltage Network Analysis Platform")
        gr.Markdown(
            "Analyze and predict voltage network behavior using advanced AI models."
        )
        
        # State variables
        state = gr.State({
            'data': None,
            'model': None
        })
        
        with gr.Tabs():
            # Data Processing Tab
            with gr.Tab("Data Processing"):
                with gr.Row():
                    file_input = gr.File(
                        label="Upload Data (CSV)",
                        file_types=[".csv"]
                    )
                    process_output = gr.Textbox(
                        label="Processing Report",
                        lines=5
                    )
                
                with gr.Row():
                    handle_missing = gr.Checkbox(
                        label="Handle Missing Values",
                        value=True
                    )
                    detect_anomalies = gr.Checkbox(
                        label="Detect Anomalies",
                        value=True
                    )
                    add_features = gr.Checkbox(
                        label="Add Engineered Features",
                        value=True
                    )
                
                process_btn = gr.Button("Process Data")
                
                def process_callback(
                    file,
                    handle_missing,
                    detect_anomalies,
                    add_features,
                    state
                ):
                    df, report = process_data(
                        file.name,
                        handle_missing,
                        detect_anomalies,
                        add_features
                    )
                    state['data'] = df
                    return report, state
                
                process_btn.click(
                    process_callback,
                    inputs=[
                        file_input,
                        handle_missing,
                        detect_anomalies,
                        add_features,
                        state
                    ],
                    outputs=[process_output, state]
                )
            
            # Model Training Tab
            with gr.Tab("Model Training"):
                with gr.Row():
                    model_type = gr.Dropdown(
                        choices=list(models_config.keys()),
                        label="Model Type"
                    )
                    epochs = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=10,
                        label="Number of Epochs"
                    )
                    batch_size = gr.Slider(
                        minimum=16,
                        maximum=128,
                        value=32,
                        step=16,
                        label="Batch Size"
                    )
                
                train_btn = gr.Button("Train Model")
                
                with gr.Row():
                    training_report = gr.Textbox(
                        label="Training Report",
                        lines=5
                    )
                    training_plot = gr.Plot(label="Training History")
                
                def train_callback(
                    model_type,
                    epochs,
                    batch_size,
                    state
                ):
                    if state['data'] is None:
                        return "Error: No data available", None, state
                    
                    model, report, plot = train_and_evaluate(
                        state['data'],
                        model_type,
                        epochs,
                        batch_size
                    )
                    state['model'] = model
                    return report, plot, state
                
                train_btn.click(
                    train_callback,
                    inputs=[
                        model_type,
                        epochs,
                        batch_size,
                        state
                    ],
                    outputs=[
                        training_report,
                        training_plot,
                        state
                    ]
                )
            
            # Predictions Tab
            with gr.Tab("Predictions"):
                predict_btn = gr.Button("Generate Predictions")
                
                with gr.Row():
                    prediction_plot = gr.Plot(label="Predictions")
                    metrics_report = gr.Textbox(
                        label="Prediction Metrics",
                        lines=5
                    )
                
                def predict_callback(state):
                    if state['model'] is None:
                        return None, "Error: No trained model available"
                    if state['data'] is None:
                        return None, "Error: No data available"
                    
                    plot, report = make_predictions(
                        state['model'],
                        state['data']
                    )
                    return plot, report
                
                predict_btn.click(
                    predict_callback,
                    inputs=[state],
                    outputs=[prediction_plot, metrics_report]
                )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    ) 