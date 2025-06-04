import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    """
    Dataset dla szeregów czasowych w estymacji napięć sieci nn.
    Przygotowuje dane w formacie sekwencji dla modeli deep learning.
    """
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def prepare_data(data, target_column, seq_length):
    """
    Przygotowuje dane do trenowania modeli estymacji napięć w sieci nn.
    
    Args:
        data: DataFrame z danymi pomiarowymi z sieci nn
        target_column: nazwa kolumny docelowej do predykcji (voltage_extreme)
        seq_length: długość sekwencji wejściowej
        
    Returns:
        X, y: przygotowane sekwencje i cele
        scaler_X, scaler_y: skalery do denormalizacji
        feature_names: nazwy cech
    """
    
    # Wybór kolumn cech dla sieci nn (bez timestamp i target)
    potential_features = [
        'current_L1', 'current_L2', 'current_L3',
        'voltage_L1', 'voltage_L2', 'voltage_L3', 
        'active_power_total', 'reactive_power_total',
        'frequency', 'voltage_thd', 'current_thd',
        'irradiance', 'pv_power', 'temperature', 'humidity',
        'line_length_total', 'line_resistance', 'pv_connections'
    ]
    
    # Sprawdzenie dostępności kolumn
    available_features = [col for col in potential_features if col in data.columns]
    
    if target_column not in available_features:
        raise ValueError(f"Kolumna docelowa '{target_column}' nie została znaleziona w danych")
    
    # Przygotowanie danych cech
    X_data = data[available_features].values
    y_data = data[target_column].values
    
    # Sprawdzenie i obsługa wartości NaN
    if np.any(np.isnan(X_data)) or np.any(np.isnan(y_data)):
        print("Wykryto wartości NaN - wypełnianie interpolacją...")
        df_temp = pd.DataFrame(X_data, columns=available_features)
        df_temp = df_temp.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        X_data = df_temp.values
        
        y_series = pd.Series(y_data)
        y_data = y_series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
    
    # Normalizacja danych
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    
    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()
    
    # Tworzenie sekwencji
    X_sequences = []
    y_sequences = []
    
    for i in range(seq_length, len(X_scaled)):
        X_sequences.append(X_scaled[i-seq_length:i])
        y_sequences.append(y_scaled[i])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    print(f"Przygotowano {len(X_sequences)} sekwencji")
    print(f"Kształt X: {X_sequences.shape}")
    print(f"Kształt y: {y_sequences.shape}")
    
    return X_sequences, y_sequences, scaler_X, scaler_y, available_features

def create_sliding_windows(data, window_size, target_col_idx):
    """
    Tworzy przesuwające się okna dla szeregów czasowych.
    
    Args:
        data: dane w formacie numpy array
        window_size: rozmiar okna
        target_col_idx: indeks kolumny docelowej
        
    Returns:
        X, y: sekwencje wejściowe i cele
    """
    X, y = [], []
    
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, target_col_idx])
    
    return np.array(X), np.array(y)

def add_technical_indicators(data):
    """
    Dodaje wskaźniki techniczne do danych szeregów czasowych.
    Przydatne dla lepszej predykcji napięć w sieci.
    
    Args:
        data: DataFrame z danymi pomiarowymi
        
    Returns:
        DataFrame z dodatkowymi wskaźnikami
    """
    df = data.copy()
    
    # Średnie kroczące
    for col in ['current', 'active_power', 'reactive_power']:
        if col in df.columns:
            df[f'{col}_ma_5'] = df[col].rolling(window=5).mean()
            df[f'{col}_ma_15'] = df[col].rolling(window=15).mean()
            df[f'{col}_ma_30'] = df[col].rolling(window=30).mean()
    
    # Odchylenia standardowe kroczące
    for col in ['current', 'active_power']:
        if col in df.columns:
            df[f'{col}_std_5'] = df[col].rolling(window=5).std()
            df[f'{col}_std_15'] = df[col].rolling(window=15).std()
    
    # Różnice pierwszego rzędu (zmiany)
    for col in ['current', 'active_power', 'reactive_power', 'frequency']:
        if col in df.columns:
            df[f'{col}_diff'] = df[col].diff()
            df[f'{col}_pct_change'] = df[col].pct_change()
    
    # Współczynnik mocy (jeśli dostępne obie moce)
    if 'active_power' in df.columns and 'reactive_power' in df.columns:
        # Zapobieganie dzieleniu przez zero
        apparent_power = np.sqrt(df['active_power']**2 + df['reactive_power']**2)
        apparent_power = np.where(apparent_power == 0, 1e-6, apparent_power)
        df['power_factor'] = df['active_power'] / apparent_power
    
    # Wskaźniki harmoniczne (na podstawie częstotliwości)
    if 'frequency' in df.columns:
        df['freq_deviation'] = df['frequency'] - 50.0  # Odchylenie od częstotliwości nominalnej
        df['freq_stability'] = df['frequency'].rolling(window=10).std()
    
    # Wskaźniki pogodowe (jeśli dostępne napromieniowanie)
    if 'irradiance' in df.columns:
        df['irradiance_ma_10'] = df['irradiance'].rolling(window=10).mean()
        df['irradiance_positive'] = (df['irradiance'] > 0).astype(int)
        
        # Klasy nasłonecznienia
        df['solar_class'] = pd.cut(
            df['irradiance'], 
            bins=[-np.inf, 0, 200, 500, 800, np.inf],
            labels=['night', 'dawn_dusk', 'cloudy', 'partial_sun', 'full_sun']
        )
        
        # One-hot encoding dla klas nasłonecznienia
        solar_dummies = pd.get_dummies(df['solar_class'], prefix='solar')
        df = pd.concat([df, solar_dummies], axis=1)
        df.drop('solar_class', axis=1, inplace=True)
    
    # Wskaźniki czasowe
    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        
        # Cykliczne enkodowanie czasu
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Wypełnienie wartości NaN
    df = df.bfill().ffill()
    
    return df

def detect_anomalies(data, columns=None, method='iqr'):
    """
    Wykrywa anomalie w danych pomiarowych sieci elektrycznej.
    
    Args:
        data: DataFrame z danymi
        columns: lista kolumn do sprawdzenia (jeśli None, sprawdza wszystkie numeryczne)
        method: metoda wykrywania ('iqr', 'zscore')
        
    Returns:
        DataFrame z dodatkową kolumną 'is_anomaly'
    """
    df = data.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    anomalies = pd.Series([False] * len(df), index=df.index)
    
    for col in columns:
        if col in df.columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_anomalies = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_anomalies = z_scores > 3
            
            anomalies = anomalies | col_anomalies
    
    df['is_anomaly'] = anomalies
    
    return df

def validate_data_quality(data):
    """
    Sprawdza jakość danych dla estymacji napięć.
    
    Args:
        data: DataFrame z danymi
        
    Returns:
        dict z informacjami o jakości danych
    """
    quality_report = {}
    
    # Podstawowe statystyki
    quality_report['total_records'] = len(data)
    quality_report['total_features'] = len(data.columns)
    
    # Sprawdzenie wartości NaN
    nan_counts = data.isnull().sum()
    quality_report['nan_counts'] = nan_counts.to_dict()
    quality_report['total_nan'] = nan_counts.sum()
    quality_report['nan_percentage'] = (nan_counts.sum() / (len(data) * len(data.columns))) * 100
    
    # Sprawdzenie duplikatów
    quality_report['duplicates'] = data.duplicated().sum()
    
    # Sprawdzenie zakresów fizycznych
    physical_checks = {}
    
    if 'current' in data.columns:
        physical_checks['current_negative'] = (data['current'] < 0).sum()
        physical_checks['current_too_high'] = (data['current'] > 1000).sum()  # Założenie: max 1000A
    
    if 'frequency' in data.columns:
        physical_checks['frequency_out_of_range'] = (
            (data['frequency'] < 45) | (data['frequency'] > 55)
        ).sum()
    
    if 'active_power' in data.columns:
        physical_checks['negative_active_power'] = (data['active_power'] < 0).sum()
    
    quality_report['physical_checks'] = physical_checks
    
    # Sprawdzenie ciągłości czasowej (jeśli jest kolumna time)
    if 'time' in data.columns:
        try:
            time_series = pd.to_datetime(data['time'])
            time_diffs = time_series.diff().dt.total_seconds()
            quality_report['time_gaps'] = (time_diffs > 900).sum()  # Gaps > 15 min
            quality_report['median_interval'] = time_diffs.median()
        except:
            quality_report['time_parsing_error'] = True
    
    return quality_report