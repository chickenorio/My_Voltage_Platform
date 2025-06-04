import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import os

def create_time_series_plot(data, column, title=None, y_label=None):
    """Tworzy wykres szeregu czasowego dla wybranej kolumny"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['datetime'], 
        y=data[column],
        name=column,
        line=dict(color='red', width=1)
    ))
    
    # Dodanie linii normatywnych dla voltage_extreme
    if column == 'voltage_extreme':
        fig.add_hline(y=253, line_dash="dash", line_color="orange", 
                     annotation_text="Górna granica (+10%)")
        fig.add_hline(y=207, line_dash="dash", line_color="blue", 
                     annotation_text="Dolna granica (-10%)")
    
    fig.update_layout(
        title=title or f"{column} w czasie",
        xaxis_title="Czas",
        yaxis_title=y_label or "Wartość",
        height=500
    )
    
    return fig

def create_correlation_matrix(data):
    """Tworzy macierz korelacji"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Macierz korelacji parametrów sieci",
            color_continuous_scale="RdBu"
        )
        return fig
    return None

def create_anomaly_plot(data, column, anomalies):
    """Tworzy wykres z zaznaczonymi anomaliami"""
    fig = go.Figure()
    
    # Dodanie normalnych punktów
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[column],
        mode='markers',
        name='Normalne wartości',
        marker=dict(color='blue', size=8)
    ))
    
    # Dodanie anomalii
    fig.add_trace(go.Scatter(
        x=anomalies.index,
        y=anomalies[column],
        mode='markers',
        name='Anomalie',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    fig.update_layout(
        title=f"Wykryte anomalie w kolumnie {column}",
        xaxis_title="Indeks",
        yaxis_title="Wartość",
        height=500
    )
    
    return fig

def create_model_comparison_plot(results):
    """Tworzy wykres porównawczy modeli"""
    fig = go.Figure()
    
    for model_name in results.keys():
        # Rzeczywiste wartości
        fig.add_trace(go.Scatter(
            y=results[model_name]['actuals'],
            name=f'Rzeczywiste - {model_name}',
            line=dict(color='blue', width=2)
        ))
        
        # Predykcje
        fig.add_trace(go.Scatter(
            y=results[model_name]['predictions'],
            name=f'Predykcje {model_name}',
            line=dict(color='red', width=2)
        ))
    
    fig.update_layout(
        title="Porównanie predykcji modeli",
        xaxis_title="Próbka",
        yaxis_title="Napięcie [V]",
        height=400
    )
    
    return fig

def create_model_radar_plot(comparison_data):
    """Tworzy wykres radarowy porównujący modele"""
    fig = go.Figure()
    
    for i, model in enumerate(comparison_data['Model']):
        fig.add_trace(go.Scatterpolar(
            r=[comparison_data.iloc[i]['Dokładność'], 
               comparison_data.iloc[i]['Szybkość treningu'], 
               comparison_data.iloc[i]['Interpretacja'], 
               comparison_data.iloc[i]['Wymagania danych'], 
               comparison_data.iloc[i]['Stabilność']],
            theta=['Dokładność', 'Szybkość treningu', 'Interpretacja', 
                  'Wymagania danych', 'Stabilność'],
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Porównanie charakterystyk modeli"
    )
    
    return fig 