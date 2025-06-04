import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from typing import Dict, List, Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.1):
        """
        Inicjalizacja detektora anomalii.
        
        Args:
            method: Metoda wykrywania anomalii ('isolation_forest', 'local_outlier_factor', 'elliptic_envelope')
            contamination: Proporcja anomalii w danych (0.0 do 0.5)
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicjalizacja modelu wykrywania anomalii."""
        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.method == 'local_outlier_factor':
            self.model = LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=20,
                novelty=True
            )
        elif self.method == 'elliptic_envelope':
            self.model = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
        else:
            raise ValueError(f"Nieznana metoda: {self.method}")
    
    def detect(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Wykrywa anomalie w danych.
        
        Args:
            data: DataFrame z danymi do analizy
            
        Returns:
            Tuple[pd.DataFrame, Dict]: DataFrame z anomaliami i statystyki
        """
        # Wybierz tylko kolumny numeryczne
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("Brak kolumn numerycznych w danych")
        
        # Wytrenuj model i przewiduj anomalie
        predictions = self.model.fit_predict(numeric_data)
        
        # Znajdź indeksy anomalii (predykcja -1 oznacza anomalię)
        anomaly_indices = np.where(predictions == -1)[0]
        
        # Utwórz DataFrame z anomaliami
        anomalies = data.iloc[anomaly_indices].copy()
        
        # Dodaj wynik anomalii (score)
        if hasattr(self.model, 'score_samples'):
            anomaly_scores = self.model.score_samples(numeric_data.iloc[anomaly_indices])
            anomalies['anomaly_score'] = anomaly_scores
        
        # Przygotuj statystyki
        stats = {
            'total_samples': len(data),
            'anomaly_count': len(anomalies),
            'anomaly_percentage': (len(anomalies) / len(data)) * 100,
            'method': self.method,
            'contamination': self.contamination
        }
        
        return anomalies, stats
    
    def get_anomaly_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generuje podsumowanie anomalii dla każdej kolumny numerycznej.
        
        Args:
            data: DataFrame z danymi
            
        Returns:
            pd.DataFrame: Podsumowanie anomalii
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        summary = []
        
        for col in numeric_cols:
            # Oblicz statystyki dla kolumny
            mean = data[col].mean()
            std = data[col].std()
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            
            # Znajdź potencjalne anomalie (metoda IQR)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            anomalies = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            summary.append({
                'column': col,
                'mean': mean,
                'std': std,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'anomaly_count': len(anomalies),
                'anomaly_percentage': (len(anomalies) / len(data)) * 100
            })
        
        return pd.DataFrame(summary)

def detect_anomalies(data: pd.DataFrame, method: str = 'isolation_forest', contamination: float = 0.1) -> pd.DataFrame:
    """
    Funkcja pomocnicza do wykrywania anomalii.
    
    Args:
        data: DataFrame z danymi
        method: Metoda wykrywania anomalii
        contamination: Proporcja anomalii
        
    Returns:
        pd.DataFrame: DataFrame z wykrytymi anomaliami
    """
    detector = AnomalyDetector(method=method, contamination=contamination)
    anomalies, _ = detector.detect(data)
    return anomalies

def remove_anomalies(data: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    """
    Usuwa wykryte anomalie z danych.
    
    Args:
        data: Oryginalny DataFrame
        anomalies: DataFrame z anomaliami do usunięcia
        
    Returns:
        pd.DataFrame: DataFrame z usuniętymi anomaliami
    """
    # Znajdź indeksy anomalii
    anomaly_indices = anomalies.index
    
    # Usuń anomalie z danych
    cleaned_data = data.drop(anomaly_indices)
    
    return cleaned_data 