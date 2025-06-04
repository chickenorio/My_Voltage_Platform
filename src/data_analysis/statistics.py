import pandas as pd
import numpy as np
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Oblicza podstawowe statystyki opisowe dla danych.
    
    Args:
        data: DataFrame z danymi
        
    Returns:
        Słownik ze statystykami
    """
    # Wybierz tylko kolumny numeryczne
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return {}
    
    # Oblicz statystyki dla każdej kolumny
    stats = {}
    for col in numeric_cols:
        col_stats = {
            'Średnia': data[col].mean(),
            'Mediana': data[col].median(),
            'Odchylenie standardowe': data[col].std(),
            'Minimum': data[col].min(),
            'Maximum': data[col].max(),
            'Kwartyl 25%': data[col].quantile(0.25),
            'Kwartyl 75%': data[col].quantile(0.75),
            'Liczba unikalnych wartości': data[col].nunique(),
            'Liczba wartości null': data[col].isnull().sum()
        }
        stats[col] = col_stats
    
    return stats

def generate_statistics_report(data: pd.DataFrame) -> str:
    """
    Generuje raport HTML ze statystykami i wizualizacjami.
    
    Args:
        data: DataFrame z danymi
        
    Returns:
        String z raportem HTML
    """
    # Wybierz tylko kolumny numeryczne
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return "<h1>Brak danych numerycznych do analizy</h1>"
    
    # Generuj HTML
    html = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            h1, h2 { color: #333; }
        </style>
    </head>
    <body>
        <h1>Raport statystyczny</h1>
    """
    
    # Dodaj statystyki dla każdej kolumny
    for col in numeric_cols:
        html += f"<h2>Statystyki dla {col}</h2>"
        html += "<table>"
        html += "<tr><th>Metryka</th><th>Wartość</th></tr>"
        
        stats = calculate_statistics(data)[col]
        for metric, value in stats.items():
            html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
        
        html += "</table>"
        
        # Dodaj wykresy
        fig = make_subplots(rows=1, cols=2,
                          subplot_titles=("Histogram", "Box Plot"))
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=data[col], name="Histogram"),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=data[col], name="Box Plot"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        
        # Dodaj wykresy do HTML
        html += f"<div>{fig.to_html(full_html=False)}</div>"
    
    html += """
    </body>
    </html>
    """
    
    return html 