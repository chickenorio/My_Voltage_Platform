# ⚡ Voltage Network Analysis Platform

A comprehensive platform for analyzing and predicting low-voltage electrical network behavior using advanced AI models.

## Features

- **Data Management**
  - Automated CSV data ingestion with versioning (DVC/Git-LFS)
  - Automatic data profiling and quality reports
  - Missing value detection and handling
  - Anomaly detection and removal
  - Feature engineering and validation

- **AI Models**
  - Set Transformer architecture for voltage prediction
  - Hyperparameter optimization with Optuna
  - Model training with MLflow tracking
  - Performance visualization and comparison

- **User Interfaces**
  - Interactive Streamlit web application
  - Alternative Gradio interface
  - Real-time visualization and analysis

## Project Structure

```
my_voltage_platform/
├── data/
│   ├── raw/          # Raw CSV files (versioned with DVC)
│   └── processed/    # Processed datasets
├── src/
│   ├── data_ingest/  # Data loading and versioning
│   ├── profiling/    # Data profiling tools
│   ├── missing_handler/ # Missing value handling
│   ├── anomaly_detector/ # Anomaly detection
│   ├── feature_engineering/ # Feature creation
│   ├── correlation/ # Correlation analysis
│   ├── data_validation/ # Data quality checks
│   ├── ai_models/
│   │   ├── config.py # Configuration loader
│   │   ├── data.py # Dataset creation
│   │   ├── train.py # Training pipeline
│   │   └── models/ # Model implementations
│   └── visualization/ # Plotting utilities
├── ui/
│   ├── streamlit_app.py # Streamlit interface
│   └── gradio_interface.py # Gradio interface
├── notebooks/
│   └── train_models.ipynb # End-to-end example
├── config/
│   ├── models.yaml # Model hyperparameters
│   └── preprocessing.yaml # Data processing settings
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/voltage-platform.git
cd voltage-platform
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize DVC for data versioning:
```bash
dvc init
dvc remote add -d storage s3://your-bucket/path  # Optional: Configure remote storage
```

## Usage

### Data Processing

1. Place your CSV files in `data/raw/`
2. Run data processing:
```bash
python src/data_ingest/data_loader.py --input data/raw/your_file.csv
```

### Model Training

1. Configure model parameters in `config/models.yaml`
2. Train models:
```bash
python src/ai_models/train.py --models set_transformer --optimize
```

### Web Interface

1. Start Streamlit app:
```bash
streamlit run ui/streamlit_app.py
```

2. Or use Gradio interface:
```bash
python ui/gradio_interface.py
```

## Model Architectures

### Set Transformer
- Multi-head attention mechanism
- Permutation-invariant processing
- Configurable architecture depth and width
- Voltage-specific loss function

## Configuration

### Model Parameters (`config/models.yaml`)
```yaml
set_transformer:
  hidden_dim: 128
  num_heads: 4
  num_layers: 3
  dropout: 0.1
  lr: 1e-4
  batch_size: 64
  epochs: 50
  device: cuda
```

### Preprocessing Settings (`config/preprocessing.yaml`)
```yaml
missing_values:
  voltage:
    strategy: interpolate
    method: time
    limit: 24  # hours
  current:
    strategy: model_impute
    method: knn
    n_neighbors: 5

feature_engineering:
  window_sizes: [6, 12, 24]
  functions: [mean, std, min, max]
```

## Development

### Adding New Models

1. Create model class in `src/ai_models/models/`
2. Inherit from `BaseVoltageModel`
3. Implement required methods
4. Add configuration to `config/models.yaml`
5. Register model in training pipeline

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- Streamlit 1.0+
- Gradio 2.8+
- pandas-profiling 4.1+
- Great Expectations 0.15+
- MLflow 1.20+
- Optuna 2.10+
- DVC 2.8+

## License

MIT License - see LICENSE file for details

## Authors

- Your Name - Initial work

## Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit team for the web app framework
- Open-source community for various tools and libraries 