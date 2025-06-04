import pandas as pd
import numpy as np
from datetime import datetime

def log_operation(operation, details):
    """Funkcja do logowania operacji na danych"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"\n[{timestamp}] === OPERACJA: {operation} ===\n"
    log_message += f"Szczegóły: {details}\n"
    log_message += "=" * 50
    print(log_message)

def test_anomaly_handling():
    # Wczytaj dane
    log_operation("TEST", "Rozpoczynam test obsługi anomalii")
    data = pd.read_csv('data/working_data.csv')
    log_operation("TEST", f"Wczytano {len(data)} rekordów")
    
    # Wybierz kolumnę do analizy
    column_to_analyze = 'voltage_extreme'
    
    # Wykryj anomalie metodą odchylenia standardowego
    mean = data[column_to_analyze].mean()
    std = data[column_to_analyze].std()
    threshold = 3.0
    mask = abs(data[column_to_analyze] - mean) > threshold * std
    anomalies = data[mask]
    
    log_operation("TEST", f"Wykryto {len(anomalies)} anomalii")
    print("\nPrzykładowe anomalie:")
    print(anomalies[[column_to_analyze]].head())
    
    # Test 1: Usuwanie anomalii
    log_operation("TEST", "Test usuwania anomalii")
    data_cleaned = data[~mask].reset_index(drop=True)
    log_operation("TEST", f"Po usunięciu zostało {len(data_cleaned)} rekordów")
    
    # Test 2: Zastępowanie wartościami granicznymi
    log_operation("TEST", "Test zastępowania wartościami granicznymi")
    data_clipped = data.copy()
    data_clipped.loc[mask & (data[column_to_analyze] < mean - threshold * std), column_to_analyze] = mean - threshold * std
    data_clipped.loc[mask & (data[column_to_analyze] > mean + threshold * std), column_to_analyze] = mean + threshold * std
    
    # Sprawdź czy wartości zostały poprawnie zastąpione
    clipped_anomalies = data_clipped[abs(data_clipped[column_to_analyze] - mean) > threshold * std]
    log_operation("TEST", f"Po zastąpieniu zostało {len(clipped_anomalies)} anomalii")
    
    # Wyświetl statystyki przed i po
    print("\nStatystyki przed:")
    print(data[column_to_analyze].describe())
    print("\nStatystyki po usunięciu:")
    print(data_cleaned[column_to_analyze].describe())
    print("\nStatystyki po zastąpieniu:")
    print(data_clipped[column_to_analyze].describe())
    
    # Zapisz wyniki do plików
    data_cleaned.to_csv('data/test_cleaned.csv', index=False)
    data_clipped.to_csv('data/test_clipped.csv', index=False)
    log_operation("TEST", "Zapisano wyniki testów do plików")

if __name__ == "__main__":
    test_anomaly_handling() 