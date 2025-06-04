import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer

def detect_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    """
    Wykrywa anomalie w danych numerycznych używając algorytmu Isolation Forest.
    
    Args:
        data (pd.DataFrame): DataFrame z danymi do analizy
        
    Returns:
        pd.DataFrame: DataFrame zawierający wykryte anomalie
    """
    # Wybierz tylko kolumny numeryczne
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Inicjalizuj i wytrenuj model Isolation Forest
    model = IsolationForest(contamination=0.1, random_state=42)
    predictions = model.fit_predict(numeric_data)
    
    # Znajdź indeksy anomalii (predykcja -1 oznacza anomalię)
    anomaly_indices = np.where(predictions == -1)[0]
    
    # Utwórz DataFrame z anomaliami
    anomalies = data.iloc[anomaly_indices].copy()
    anomalies['anomaly_score'] = model.score_samples(numeric_data.iloc[anomaly_indices])
    
    return anomalies

def remove_anomalies(data: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    """
    Usuwa wykryte anomalie z danych.
    
    Args:
        data (pd.DataFrame): Oryginalny DataFrame
        anomalies (pd.DataFrame): DataFrame zawierający anomalie do usunięcia
        
    Returns:
        pd.DataFrame: DataFrame z usuniętymi anomaliami
    """
    # Znajdź indeksy anomalii
    anomaly_indices = anomalies.index
    
    # Usuń anomalie z danych
    cleaned_data = data.drop(anomaly_indices)
    
    return cleaned_data

def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Wypełnia brakujące wartości w danych używając algorytmu KNN.
    
    Args:
        data (pd.DataFrame): DataFrame z brakującymi wartościami
        
    Returns:
        pd.DataFrame: DataFrame z wypełnionymi brakującymi wartościami
    """
    # Wybierz tylko kolumny numeryczne
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Inicjalizuj i wytrenuj imputer KNN
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(numeric_data)
    
    # Utwórz nowy DataFrame z wypełnionymi wartościami
    filled_data = pd.DataFrame(imputed_data, columns=numeric_data.columns, index=numeric_data.index)
    
    # Zachowaj oryginalne kolumny nienumeryczne
    non_numeric_data = data.select_dtypes(exclude=[np.number])
    result = pd.concat([filled_data, non_numeric_data], axis=1)
    
    return result 