import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple

def create_correlation_matrix(data: pd.DataFrame) -> go.Figure:
    """
    Tworzy macierz korelacji dla danych numerycznych.
    
    Args:
        data: DataFrame z danymi
        
    Returns:
        Plotly Figure z macierzą korelacji
    """
    # Wybierz tylko kolumny numeryczne
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        # Oblicz macierz korelacji
        corr_matrix = data[numeric_cols].corr()
        
        # Utwórz wykres
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Macierz korelacji parametrów sieci",
            color_continuous_scale="RdBu"
        )
        
        # Dostosuj układ
        fig.update_layout(
            height=800,
            width=1000,
            xaxis_title="Parametry",
            yaxis_title="Parametry"
        )
        
        return fig
    return None

def analyze_correlations(data: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
    """
    Analizuje korelacje między parametrami i zwraca najsilniejsze zależności.
    
    Args:
        data: DataFrame z danymi
        
    Returns:
        Słownik z najsilniejszymi korelacjami dla każdego parametru
    """
    # Wybierz tylko kolumny numeryczne
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) <= 1:
        return {}
    
    # Oblicz macierz korelacji
    corr_matrix = data[numeric_cols].corr()
    
    # Znajdź najsilniejsze korelacje dla każdego parametru
    correlations = {}
    for col in numeric_cols:
        # Pobierz korelacje dla danej kolumny
        col_correlations = corr_matrix[col].drop(col)
        
        # Znajdź 3 najsilniejsze korelacje (pozytywne i negatywne)
        strongest = col_correlations.abs().nlargest(3)
        
        # Zachowaj znak korelacji
        correlations[col] = [
            (idx, col_correlations[idx])
            for idx in strongest.index
        ]
    
    return correlations 