import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import io
import base64
from datetime import datetime
import warnings
import os
import sys

# Dodaj ścieżki do modułów
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# Funkcja log_operation - przeniesiona na początek pliku

def log_operation(operation, details, function_name=None, button_name=None, data_stats=None):
    """Funkcja do logowania operacji na danych z rozszerzonymi informacjami diagnostycznymi"""
    # Tworzymy folder logs jeśli nie istnieje
    if not os.path.exists('data/logs'):
        os.makedirs('data/logs')
    # Generujemy nazwę pliku z datą
    current_date = datetime.now().strftime('%Y-%m-%d')
    log_file = f'data/logs/app_log_{current_date}.txt'
    # Formatujemy wiadomość z czasem
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"\n[{timestamp}] === OPERACJA: {operation} ===\n"
    # Dodajemy informacje o funkcji
    if function_name:
        log_message += f"Funkcja: {function_name}\n"
    # Dodajemy informacje o przycisku
    if button_name:
        log_message += f"Przycisk: {button_name}\n"
    # Dodajemy szczegóły operacji
    log_message += f"Szczegóły: {details}\n"
    # Dodajemy statystyki danych jeśli dostępne
    if data_stats:
        log_message += f"Statystyki danych:\n{data_stats}\n"
    log_message += "=" * 50
    # Zapisujemy do pliku
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message)
    # Wyświetlamy też w konsoli
    print(log_message)

from utils.data_manager import load_working_data, save_working_data, initialize_working_data
from data_analysis.correlation import create_correlation_matrix, analyze_correlations
from data_analysis.statistics import calculate_statistics, generate_statistics_report
from data_analysis.data_processing import fill_missing_values
from anomaly_detector import detect_anomalies, remove_anomalies
from missing_handler.missing_handler import MissingHandler

warnings.filterwarnings('ignore')

# Konfiguracja strony
st.set_page_config(
    page_title="Estymator Stanu Sieci Niskiego Napięcia",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Główny tytuł
st.title("⚡ Estymator Stanu Sieci Niskiego Napięcia")
st.markdown("### System sztucznej inteligencji do estymacji skrajnych napięć w sieciach nn")

# Menu boczne
st.sidebar.title("Menu")

# Główne zakładki
tab_main, tab_anomalies, tab_models, tab_about = st.tabs(["🏠 Estymacja napięć", "🔍 Obsługa braków i anomalii", "🧠 Modele AI", "📚 O systemie"])

with tab_anomalies:
    st.header("🔍 Obsługa braków i anomalii")
    st.markdown("""
    W tej sekcji możesz wykryć i usunąć anomalie oraz wypełnić braki w danych.
    """)
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        # Inicjalizacja MissingHandler
        if 'missing_handler' not in st.session_state:
            st.session_state['missing_handler'] = MissingHandler()
        
        # Sekcja analizy braków
        st.subheader("📊 Analiza braków w danych")
        missing_stats = st.session_state['missing_handler'].get_missing_stats(data)
        
        # Wyświetlanie statystyk braków
        if missing_stats['columns_with_missing']:
            st.write("### Kolumny z brakującymi wartościami:")
            missing_df = pd.DataFrame({
                'Kolumna': list(missing_stats['columns_with_missing'].keys()),
                'Liczba braków': list(missing_stats['columns_with_missing'].values()),
                'Procent braków': list(missing_stats['missing_percentages'].values())
            })
            st.dataframe(missing_df.style.format({'Procent braków': '{:.2f}%'}))
            
            # Wybór metody wypełniania braków
            st.subheader("🔄 Wypełnianie braków")
            method = st.selectbox(
                "Wybierz metodę wypełniania braków:",
                ['interpolate', 'forward_fill', 'backward_fill', 'model_impute'],
                format_func=lambda x: {
                    'interpolate': 'Interpolacja liniowa',
                    'forward_fill': 'Wypełnienie wprzód',
                    'backward_fill': 'Wypełnienie wstecz',
                    'model_impute': 'Model uczenia maszynowego'
                }[x]
            )
            
            if st.button("🔄 Wypełnij braki"):
                log_operation("WYPEŁNIANIE BRAKÓW", f"Rozpoczynam wypełnianie braków metodą {method}")
                for column in missing_stats['columns_with_missing'].keys():
                    data = st.session_state['missing_handler'].apply_config_strategy(data, column, method)
                st.session_state['data'] = data
                st.success("✅ Braki zostały wypełnione")
                
                # Wyświetl podsumowanie operacji
                summary = st.session_state['missing_handler'].get_imputation_summary()
                st.write("### Podsumowanie operacji wypełniania:")
                st.dataframe(summary)
        else:
            st.success("✅ Brak wartości brakujących w danych!")
        
        # Sekcja anomalii
        st.subheader("🔍 Wykrywanie anomalii")
        
        # Wybór metody wykrywania anomalii
        anomaly_method = st.selectbox(
            "Wybierz metodę wykrywania anomalii:",
            ['z_score', 'iqr', 'isolation_forest', 'local_outlier_factor', 'elliptic_envelope'],
            format_func=lambda x: {
                'z_score': 'Z-score (odchylenie standardowe)',
                'iqr': 'IQR (kwartyle)',
                'isolation_forest': 'Isolation Forest',
                'local_outlier_factor': 'Local Outlier Factor',
                'elliptic_envelope': 'Elliptic Envelope'
            }[x]
        )
        
        # Parametry dla metod statystycznych
        if anomaly_method in ['z_score', 'iqr']:
            col1, col2 = st.columns(2)
            with col1:
                if anomaly_method == 'z_score':
                    threshold = st.number_input(
                        "Próg odchylenia standardowego:",
                        min_value=1.0,
                        max_value=5.0,
                        value=3.0,
                        step=0.1,
                        help="Wartości poza tym zakresem będą uznane za anomalie"
                    )
                else:  # iqr
                    threshold = st.number_input(
                        "Mnożnik IQR:",
                        min_value=1.0,
                        max_value=3.0,
                        value=1.5,
                        step=0.1,
                        help="Mnożnik zakresu międzykwartylowego"
                    )
            with col2:
                selected_columns = st.multiselect(
                    "Wybierz kolumny do analizy:",
                    data.select_dtypes(include=[np.number]).columns.tolist(),
                    default=data.select_dtypes(include=[np.number]).columns.tolist()
                )
        else:
            # Parametry dla metod uczenia maszynowego
            contamination = st.slider(
                "Proporcja anomalii w danych:",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Oczekiwana proporcja anomalii w danych (0.01 - 0.5)"
            )
        
        if st.button("🔍 Wykryj anomalie"):
            log_operation("DETEKCJA ANOMALII", f"Rozpoczynam wykrywanie anomalii metodą {anomaly_method}", button_name="Wykryj anomalie")
            
            if anomaly_method in ['z_score', 'iqr']:
                # Wykrywanie anomalii metodami statystycznymi
                anomalies = pd.DataFrame()
                stats = {
                    'total_samples': len(data),
                    'anomaly_count': 0,
                    'anomaly_percentage': 0,
                    'method': anomaly_method,
                    'threshold': threshold
                }
                
                for column in selected_columns:
                    if anomaly_method == 'z_score':
                        # Metoda Z-score
                        mean = data[column].mean()
                        std = data[column].std()
                        z_scores = np.abs((data[column] - mean) / std)
                        column_anomalies = data[z_scores > threshold]
                    else:
                        # Metoda IQR
                        q1 = data[column].quantile(0.25)
                        q3 = data[column].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                        column_anomalies = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
                    
                    anomalies = pd.concat([anomalies, column_anomalies])
                
                # Usunięcie duplikatów
                anomalies = anomalies.drop_duplicates()
                
                # Aktualizacja statystyk
                stats['anomaly_count'] = len(anomalies)
                stats['anomaly_percentage'] = (len(anomalies) / len(data)) * 100
                
                # Generowanie podsumowania
                summary = []
                for column in selected_columns:
                    if anomaly_method == 'z_score':
                        mean = data[column].mean()
                        std = data[column].std()
                        z_scores = np.abs((data[column] - mean) / std)
                        column_anomalies = data[z_scores > threshold]
                        summary.append({
                            'column': column,
                            'mean': mean,
                            'std': std,
                            'threshold': threshold,
                            'anomaly_count': len(column_anomalies),
                            'anomaly_percentage': (len(column_anomalies) / len(data)) * 100
                        })
                    else:
                        q1 = data[column].quantile(0.25)
                        q3 = data[column].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                        column_anomalies = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
                        summary.append({
                            'column': column,
                            'q1': q1,
                            'q3': q3,
                            'iqr': iqr,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'anomaly_count': len(column_anomalies),
                            'anomaly_percentage': (len(column_anomalies) / len(data)) * 100
                        })
                
                summary_df = pd.DataFrame(summary)
            else:
                # Wykrywanie anomalii metodami uczenia maszynowego
                detector = AnomalyDetector(method=anomaly_method, contamination=contamination)
                anomalies, stats = detector.detect(data)
                summary_df = detector.get_anomaly_summary(data)
            
            st.session_state['anomalies'] = anomalies
            
            # Wyświetlanie statystyk
            st.success(f"✅ Wykryto {stats['anomaly_count']} anomalii ({stats['anomaly_percentage']:.2f}% danych)")
            
            # Wyświetlanie szczegółowego podsumowania
            st.write("### Szczegółowe podsumowanie anomalii:")
            st.dataframe(summary_df)
            
            # Wyświetlanie wykrytych anomalii
            st.write("### Wykryte anomalie:")
            st.dataframe(anomalies)
        
        if st.button("🗑️ Usuń anomalie"):
            log_operation("USUWANIE ANOMALII", "Rozpoczynam usuwanie anomalii", button_name="Usuń anomalie")
            if 'anomalies' in st.session_state:
                data = remove_anomalies(data, st.session_state['anomalies'])
                st.session_state['data'] = data
                # Zapisz zaktualizowane dane do pliku working_data.csv
                data.to_csv('data/working_data.csv', index=False)
                log_operation("USUWANIE ANOMALII", "Dane po usunięciu anomalii zapisane do working_data.csv")
                st.success("✅ Anomalie zostały usunięte i zapisane do pliku")
            else:
                st.error("❌ Najpierw wykryj anomalie")
    else:
        st.info("👆 Najpierw wczytaj dane w zakładce 'Estymacja napięć'")

with tab_models:
    st.header("🧠 Architektury modeli sztucznej inteligencji")
    st.markdown("Poznaj szczegóły modeli używanych do estymacji napięć w sieciach niskiego napięcia")
    
    # Wybór modelu do opisu
    model_choice = st.selectbox(
        "Wybierz model do szczegółowego opisu:",
        ["LSTM", "PINN", "Transformer", "Hybrid"],
        key="model_desc"
    )
    
    if model_choice == "LSTM":
        st.subheader("🔄 LSTM (Long Short-Term Memory)")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **LSTM** to zaawansowana architektura sieci rekurencyjnej, specjalnie zaprojektowana 
            do przetwarzania szeregów czasowych i sekwencji danych.
            
            **Zastosowanie w estymacji napięć:**
            - Przewiduje skrajne napięcia na podstawie historycznych pomiarów
            - Uczy się wzorców obciążenia i generacji w różnych porach dnia
            - Dostrzega sezonowe zmiany w zachowaniu sieci
            
            **Zalety:**
            - ✅ Doskonale radzi sobie z długimi sekwencjami czasowymi
            - ✅ Pamięta długoterminowe zależności w danych
            - ✅ Stabilny podczas treningu
            - ✅ Sprawdzony w zastosowaniach energetycznych
            
            **Wady:**
            - ❌ Wymaga dużo danych treningowych
            - ❌ Wolniejszy niż modele feedforward
            - ❌ Nie uwzględnia bezpośrednio praw fizyki
            """)
            
        with col2:
            # Diagram LSTM
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[1, 2, 3, 4],
                y=[1, 1, 1, 1],
                mode='markers+text',
                marker=dict(size=40, color='lightblue'),
                text=['Input', 'LSTM', 'LSTM', 'Output'],
                textposition="middle center"
            ))
            fig.add_trace(go.Scatter(
                x=[1, 2, 2, 3, 3, 4],
                y=[1, 1, 1, 1, 1, 1],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
            fig.update_layout(
                title="Architektura LSTM",
                showlegend=False,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                height=200
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif model_choice == "PINN":
        st.subheader("⚗️ PINN (Physics-Informed Neural Networks)")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **PINN** to rewolucyjna architektura łącząca uczenie maszynowe z prawami fizyki,
            szczególnie przydatna w systemach energetycznych.
            
            **Zastosowanie w estymacji napięć:**
            - Uwzględnia prawa Kirchhoffa i równania przepływu mocy
            - Respektuje fizyczne ograniczenia sieci elektrycznej
            - Zapewnia fizycznie sensowne predykcje napięć
            
            **Zalety:**
            - ✅ Uwzględnia prawa fizyki w predykcjach
            - ✅ Wymaga mniej danych niż tradycyjne modele
            - ✅ Generuje fizycznie poprawne wyniki
            - ✅ Lepsze ekstrapolowanie poza dane treningowe
            
            **Wady:**
            - ❌ Bardziej złożona implementacja
            - ❌ Wymaga znajomości fizyki systemu
            - ❌ Dłuższy czas treningu
            """)
            
        with col2:
            # Diagram PINN
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[1, 2, 3],
                y=[1, 1, 1],
                mode='markers+text',
                marker=dict(size=40, color='lightgreen'),
                text=['Data', 'Neural Net', 'Output'],
                textposition="middle center"
            ))
            fig.add_trace(go.Scatter(
                x=[2],
                y=[0.5],
                mode='markers+text',
                marker=dict(size=30, color='red'),
                text=['Physics'],
                textposition="middle center"
            ))
            fig.update_layout(
                title="Architektura PINN",
                showlegend=False,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                height=200
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif model_choice == "Transformer":
        st.subheader("🎯 Transformer z mechanizmem uwagi")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Transformer** to najnowocześniejsza architektura z mechanizmem uwagi,
            która rewolucjonizuje przetwarzanie sekwencji.
            
            **Zastosowanie w estymacji napięć:**
            - Skupia uwagę na najważniejszych momentach w historii
            - Równoległy proces analizy całej sekwencji
            - Doskonale radzi sobie z długimi zależnościami
            
            **Zalety:**
            - ✅ Mechanizm uwagi na kluczowe momenty
            - ✅ Równoległy processing - szybszy trening
            - ✅ Doskonałe dla długich sekwencji
            - ✅ State-of-the-art w wielu zastosowaniach
            
            **Wady:**
            - ❌ Wymaga bardzo dużo danych
            - ❌ Wysokie wymagania obliczeniowe
            - ❌ Może przeuczyć się na małych zbiorach
            """)
            
        with col2:
            # Diagram Transformer
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[1, 2, 3, 4],
                y=[1, 1.2, 0.8, 1],
                mode='markers+text',
                marker=dict(size=35, color='orange'),
                text=['Input', 'Attention', 'Encoder', 'Output'],
                textposition="middle center"
            ))
            fig.update_layout(
                title="Architektura Transformer",
                showlegend=False,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                height=200
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif model_choice == "Hybrid":
        st.subheader("🔗 Model hybrydowy (Ensemble)")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Model hybrydowy** łączy mocne strony różnych architektur w jeden
            potężny system predykcyjny.
            
            **Zastosowanie w estymacji napięć:**
            - Kombinuje LSTM (pamięć) z PINN (fizyka)
            - Używa ensemble learning dla lepszej dokładności
            - Redukuje ryzyko błędów pojedynczych modeli
            
            **Zalety:**
            - ✅ Łączy mocne strony różnych podejść
            - ✅ Wyższa dokładność niż pojedyncze modele
            - ✅ Większa odporność na błędy
            - ✅ Lepsze uogólnianie
            
            **Wady:**
            - ❌ Większa złożoność obliczeniowa
            - ❌ Trudniejsza interpretacja wyników
            - ❌ Więcej parametrów do optymalizacji
            """)
            
        with col2:
            # Diagram Hybrid
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[1, 2, 2, 3],
                y=[1, 1.3, 0.7, 1],
                mode='markers+text',
                marker=dict(size=30, color=['blue', 'green', 'green', 'purple']),
                text=['Input', 'LSTM', 'PINN', 'Combined'],
                textposition="middle center"
            ))
            fig.update_layout(
                title="Architektura Hybrid",
                showlegend=False,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                height=200
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Porównanie modeli
    st.subheader("📊 Porównanie modeli")
    
    comparison_data = {
        'Model': ['LSTM', 'PINN', 'Transformer', 'Hybrid'],
        'Dokładność': [85, 88, 90, 95],
        'Szybkość treningu': [70, 60, 50, 40],
        'Interpretacja': [60, 90, 40, 50],
        'Wymagania danych': [80, 60, 95, 85],
        'Stabilność': [85, 95, 70, 90]
    }
    
    df_comp = pd.DataFrame(comparison_data)
    
    fig = go.Figure()
    
    for i, model in enumerate(df_comp['Model']):
        fig.add_trace(go.Scatterpolar(
            r=[df_comp.iloc[i]['Dokładność'], df_comp.iloc[i]['Szybkość treningu'], 
               df_comp.iloc[i]['Interpretacja'], df_comp.iloc[i]['Wymagania danych'], 
               df_comp.iloc[i]['Stabilność']],
            theta=['Dokładność', 'Szybkość treningu', 'Interpretacja', 'Wymagania danych', 'Stabilność'],
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
    
    st.plotly_chart(fig, use_container_width=True)

with tab_about:
    st.header("📚 O systemie estymacji napięć")
    
    st.markdown("""
    ## 🎯 Cel systemu
    
    System został opracowany w oparciu o najnowsze badania w dziedzinie estymacji napięć 
    w sieciach niskiego napięcia z wykorzystaniem sztucznej inteligencji.
    
    ## 🔬 Podstawy naukowe
    
    **Problem:** Wraz z integracją niesterowalnych źródeł energii (mikroinstalacje PV), 
    nastąpiło znaczące nasycenie sieci nn generacjami, które nie są bezpośrednio monitorowane 
    przez Operatora Systemu Dystrybucyjnego.
    
    **Rozwiązanie:** Estymator napięć oparty na AI, który wykorzystuje dane z pomiarów 
    elektrycznych w stacji SN/nn do przewidywania skrajnych wartości napięcia w krytycznych 
    punktach sieci.
    
    ## 📊 Dane wejściowe
    
    System analizuje następujące parametry:
    - **Prądy fazowe** (L1, L2, L3) - pomiary z obwodów odejściowych
    - **Napięcia fazowe** - pomiary na szynach stacji SN/nn  
    - **Moce czynna i bierna** - całkowita moc w stacji
    - **Częstotliwość sieci** - stabilność systemu elektroenergetycznego
    - **Harmoniczne** - jakość energii (THD)
    - **Dane generacji PV** - moc i napromieniowanie
    - **Dane meteorologiczne** - temperatura, wilgotność
    - **Parametry topologiczne** - długości linii, rezystancje
    
    ## 🎯 Cel predykcji
    
    **SKRAJNE NAPIĘCIE** - główny parametr estymowany przez system:
    - Najwyższe i najniższe napięcia w sieci nn
    - Napięcia w punktach krytycznych (końce linii, przyłącza PV)
    - Podstawa do regulacji napięcia w stacji
    
    ## ⚡ Zastosowania
    
    - **Regulacja napięcia** - optymalne ustawienie przełącznika zaczepów
    - **Monitorowanie sieci** - wykrywanie przekroczeń normatywnych  
    - **Planowanie** - analiza wpływu nowych przyłączy PV
    - **Eksploatacja** - wspomaganie decyzji operatora sieci
    """)

with tab_main:
    # Sidebar - konfiguracja
    st.sidebar.header("🔧 Konfiguracja")
    
    # Przycisk do wczytania przykładowych danych
    if st.sidebar.button("📊 Wczytaj przykładowe dane"):
        try:
            log_operation("WCZYTYWANIE PRZYKŁADOWYCH DANYCH", "Sprawdzanie istnienia pliku")
            if not os.path.exists('data/working_data.csv'):
                log_operation("INICJALIZACJA DANYCH", "Plik nie istnieje, rozpoczynam inicjalizację")
                if initialize_working_data():
                    data = pd.read_csv('data/working_data.csv')
                    if data is not None:
                        st.session_state['data'] = data
                        log_operation("INICJALIZACJA DANYCH", f"Zainicjalizowano {len(data)} rekordów")
                        st.sidebar.success(f"✅ Wczytano {len(data)} rekordów przykładowych")
                    else:
                        log_operation("BŁĄD INICJALIZACJI", "Nie można odczytać zainicjalizowanych danych")
                        st.sidebar.error("❌ Nie można wczytać danych")
                else:
                    log_operation("BŁĄD INICJALIZACJI", "Nie można zainicjalizować danych")
                    st.sidebar.error("❌ Nie można zainicjalizować danych")
            else:
                log_operation("WCZYTYWANIE ISTNIEJĄCYCH DANYCH", "Plik już istnieje, odczytuję dane")
                data = pd.read_csv('data/working_data.csv')
                if data is not None:
                    st.session_state['data'] = data
                    log_operation("WCZYTYWANIE DANYCH", f"Wczytano {len(data)} rekordów")
                    st.sidebar.success(f"✅ Wczytano {len(data)} rekordów")
                else:
                    log_operation("BŁĄD ODCZYTU", "Nie można odczytać danych")
                    st.sidebar.error("❌ Nie można wczytać danych")
        except Exception as e:
            log_operation("BŁĄD WCZYTYWANIA", f"Wystąpił błąd: {str(e)}")
            st.sidebar.error(f"❌ Błąd: {str(e)}")
    
    # Upload pliku
    uploaded_file = st.sidebar.file_uploader(
        "Wybierz plik CSV z danymi sieci:",
        type=['csv'],
        help="Plik powinien zawierać pomiary z stacji SN/nn oraz dane generacji PV",
        key="uploader"
    )
    
    # Przyciski do obsługi plików
    if uploaded_file is not None:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("📥 Odczytaj plik", type="primary"):
                log_operation("WCZYTYWANIE PLIKU", f"Wykryto nowy plik: {uploaded_file.name}")
                try:
                    data = pd.read_csv(uploaded_file)
                    log_operation("WCZYTYWANIE PLIKU", f"Odczytano {len(data)} rekordów z pliku")
                    
                    if save_working_data(data):
                        log_operation("ZAPIS PLIKU", "Dane zapisane do working_data.csv")
                        data = pd.read_csv('data/working_data.csv')
                        if data is not None:
                            st.session_state['data'] = data
                            log_operation("AKTUALIZACJA STANU", "Dane zaktualizowane w session_state")
                            st.sidebar.success(f"✅ Wczytano {len(data)} rekordów")
                        else:
                            log_operation("BŁĄD ODCZYTU", "Nie można odczytać zapisanych danych")
                            st.error("❌ Nie można wczytać danych")
                    else:
                        log_operation("BŁĄD ZAPISU", "Nie można zapisać danych do pliku")
                        st.error("❌ Nie można zapisać danych")
                except Exception as e:
                    log_operation("BŁĄD WCZYTYWANIA", f"Wystąpił błąd: {str(e)}")
                    st.error(f"❌ Błąd wczytywania pliku: {str(e)}")
                    st.stop()
        with col2:
            if st.button("🔄 Przeładuj dane", type="primary"):
                try:
                    log_operation("PRZEŁADOWANIE DANYCH", "Rozpoczynam odczyt pliku working_data.csv")
                    reloaded_data = pd.read_csv('data/working_data.csv')
                    log_operation("PRZEŁADOWANIE DANYCH", f"Odczytano {len(reloaded_data)} rekordów")
                    st.session_state['data'] = reloaded_data.copy()
                    log_operation("PRZEŁADOWANIE DANYCH", "Dane skopiowane do session_state")
                    st.sidebar.success("✅ Dane zostały przeładowane!")
                except Exception as e:
                    log_operation("BŁĄD PRZEŁADOWANIA", f"Wystąpił błąd: {str(e)}")
                    st.error(f"❌ Błąd przeładowania danych: {str(e)}")
    else:
        st.sidebar.info("👆 Wybierz plik aby wczytać dane")
    
    # Sprawdzenie czy dane są w session_state
    if 'data' not in st.session_state:
        st.info("👆 Wybierz plik CSV z danymi sieci lub wczytaj przykładowe dane")
        st.markdown("""
        ### 📋 Format danych wejściowych
        
        **Wymagane kolumny:**
        - `timestamp` - znacznik czasowy pomiarów
        - `voltage_extreme` - skrajne napięcie w sieci [V] (cel predykcji)
        
        ### 🎯 Cel systemu
        
        System estymuje **skrajne napięcia** w punktach krytycznych sieci nn bazując na 
        pomiarach z stacji SN/nn oraz danych generacji fotowoltaicznej.
        """)
        st.stop()
    
    data = st.session_state['data']
    
    # Sprawdzenie kolumn
    required_columns = ['timestamp', 'voltage_extreme']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        st.error(f"❌ Brakujące kluczowe kolumny: {missing_columns}")
        st.stop()
    
    # Cel predykcji
    target_column = 'voltage_extreme'
    st.sidebar.info(f"🎯 Cel predykcji: **{target_column}** (skrajne napięcie w sieci)")
    
    # Parametry modelu
    st.sidebar.subheader("⚙️ Parametry modeli")
    
    # Inicjalizacja parametrów w session_state
    if 'params' not in st.session_state:
        st.session_state['params'] = {
            'seq_length': 50,
            'batch_size': 32,
            'epochs': 30,
            'learning_rate': 0.001,
            'hidden_size': 64,
            'num_layers': 2
        }
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.session_state['params']['seq_length'] = st.number_input(
            "Długość sekwencji:", 
            min_value=10, max_value=200, 
            value=st.session_state['params']['seq_length'], 
            step=10
        )
        st.session_state['params']['batch_size'] = st.number_input(
            "Batch size:", 
            min_value=8, max_value=128, 
            value=st.session_state['params']['batch_size'], 
            step=8
        )
        st.session_state['params']['epochs'] = st.number_input(
            "Liczba epok:", 
            min_value=5, max_value=200, 
            value=st.session_state['params']['epochs'], 
            step=5
        )
    
    with col2:
        st.session_state['params']['learning_rate'] = st.number_input(
            "Learning rate:", 
            min_value=0.0001, max_value=0.01, 
            value=st.session_state['params']['learning_rate'], 
            step=0.0001, format="%.4f"
        )
        st.session_state['params']['hidden_size'] = st.number_input(
            "Hidden size:", 
            min_value=16, max_value=256, 
            value=st.session_state['params']['hidden_size'], 
            step=16
        )
        st.session_state['params']['num_layers'] = st.number_input(
            "Liczba warstw:", 
            min_value=1, max_value=5, 
            value=st.session_state['params']['num_layers'], 
            step=1
        )
    
    # Wybór modeli do trenowania
    st.sidebar.subheader("🧠 Wybór modeli")
    
    if 'selected_models' not in st.session_state:
        st.session_state['selected_models'] = ['LSTM', 'PINN']
    
    models_to_train = []
    if st.sidebar.checkbox("LSTM", value='LSTM' in st.session_state['selected_models']):
        models_to_train.append("LSTM")
    if st.sidebar.checkbox("PINN", value='PINN' in st.session_state['selected_models']):
        models_to_train.append("PINN")
    if st.sidebar.checkbox("Transformer", value='Transformer' in st.session_state['selected_models']):
        models_to_train.append("Transformer")
    if st.sidebar.checkbox("Hybrid", value='Hybrid' in st.session_state['selected_models']):
        models_to_train.append("Hybrid")
    
    st.session_state['selected_models'] = models_to_train
    
    # Przycisk trenowania
    train_button = st.sidebar.button("🚀 Rozpocznij trening", type="primary")
    
    # Przycisk resetowania danych do stanu początkowego
    if st.sidebar.button("🔄 Resetuj dane do oryginału"):
        try:
            log_operation("RESET DANYCH", "Rozpoczynam reset do oryginalnych danych")
            data = pd.read_csv('data/network_voltage_data.csv')
            log_operation("RESET DANYCH", f"Odczytano {len(data)} rekordów z oryginalnego pliku")
            data.to_csv('data/working_data.csv', index=False)
            log_operation("RESET DANYCH", "Zapisano dane do working_data.csv")
            data = pd.read_csv('data/working_data.csv')
            st.session_state['data'] = data
            log_operation("RESET DANYCH", "Zaktualizowano session_state")
            st.sidebar.success("✅ Dane zresetowane do oryginału!")
        except Exception as e:
            log_operation("BŁĄD RESETU", f"Wystąpił błąd: {str(e)}")
            st.sidebar.error(f"❌ Błąd resetowania danych: {str(e)}")
    
    # Główna część aplikacji
    if not train_button and 'training_results' not in st.session_state:
        # Analiza danych
        st.header("📊 Analiza danych sieci niskiego napięcia")
        
        # Podstawowe statystyki
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Liczba pomiarów", f"{len(data):,}")
        with col2:
            if 'timestamp' in data.columns:
                data['datetime'] = pd.to_datetime(data['timestamp'])
                start_time = data['datetime'].min()
                end_time = data['datetime'].max()
                period = (end_time - start_time).days
                st.metric("Okres [dni]", f"{period}")
            else:
                st.metric("Okres", "N/A")
        with col3:
            st.metric("Średnie napięcie", f"{data['voltage_extreme'].mean():.1f} V")
        with col4:
            voltage_violations = ((data['voltage_extreme'] < 207) | (data['voltage_extreme'] > 253)).sum()
            st.metric("Przekroczenia ±10%", f"{voltage_violations}")
        
        # Wybór typu analizy
        analysis_type = st.radio(
            "Wybierz typ analizy",
            ["Szeregi czasowe", "Korelacje", "Statystyki opisowe", "Analiza rozkładów"]
        )
        
        if analysis_type == "Szeregi czasowe":
            # Główny wykres napięć skrajnych
            data['datetime'] = pd.to_datetime(data['timestamp'])
            
            # Lista dostępnych kolumn numerycznych
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Wybór kolumny do wyświetlenia
            selected_column = st.selectbox(
                "Wybierz parametr do wyświetlenia:",
                numeric_columns,
                index=numeric_columns.index('voltage_extreme') if 'voltage_extreme' in numeric_columns else 0
            )
            
            # Konfiguracja wykresu
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['datetime'], 
                y=data[selected_column],
                name=selected_column,
                line=dict(color='red', width=1)
            ))
            
            # Dodanie linii normatywnych tylko dla voltage_extreme
            if selected_column == 'voltage_extreme':
                fig.add_hline(y=253, line_dash="dash", line_color="orange", 
                             annotation_text="Górna granica (+10%)")
                fig.add_hline(y=207, line_dash="dash", line_color="blue", 
                             annotation_text="Dolna granica (-10%)")
            
            # Konfiguracja tytułu i osi
            column_names = {
                'voltage_extreme': 'Skrajne napięcie',
                'current_L1': 'Prąd fazy L1',
                'current_L2': 'Prąd fazy L2',
                'current_L3': 'Prąd fazy L3',
                'voltage_L1': 'Napięcie fazy L1',
                'voltage_L2': 'Napięcie fazy L2',
                'voltage_L3': 'Napięcie fazy L3',
                'active_power_total': 'Moc czynna całkowita',
                'reactive_power_total': 'Moc bierna całkowita',
                'frequency': 'Częstotliwość',
                'voltage_thd': 'THD napięcia',
                'current_thd': 'THD prądu',
                'irradiance': 'Napromieniowanie',
                'pv_power': 'Moc PV',
                'temperature': 'Temperatura',
                'humidity': 'Wilgotność',
                'line_length_total': 'Całkowita długość linii',
                'line_resistance': 'Rezystancja linii',
                'pv_connections': 'Liczba przyłączy PV'
            }
            
            # Ustawienie tytułu i etykiet osi
            title = column_names.get(selected_column, selected_column)
            y_label = {
                'voltage_extreme': 'Napięcie [V]',
                'current_L1': 'Prąd [A]',
                'current_L2': 'Prąd [A]',
                'current_L3': 'Prąd [A]',
                'voltage_L1': 'Napięcie [V]',
                'voltage_L2': 'Napięcie [V]',
                'voltage_L3': 'Napięcie [V]',
                'active_power_total': 'Moc [kW]',
                'reactive_power_total': 'Moc [kVAr]',
                'frequency': 'Częstotliwość [Hz]',
                'voltage_thd': 'THD [%]',
                'current_thd': 'THD [%]',
                'irradiance': 'Napromieniowanie [W/m²]',
                'pv_power': 'Moc [kW]',
                'temperature': 'Temperatura [°C]',
                'humidity': 'Wilgotność [%]',
                'line_length_total': 'Długość [m]',
                'line_resistance': 'Rezystancja [Ω]',
                'pv_connections': 'Liczba przyłączy'
            }.get(selected_column, 'Wartość')
            
            fig.update_layout(
                title=f"{title} w czasie",
                xaxis_title="Czas",
                yaxis_title=y_label,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Dodanie statystyk pod wykresem
            st.subheader("📊 Statystyki")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Średnia", f"{data[selected_column].mean():.2f}")
            with col2:
                st.metric("Minimum", f"{data[selected_column].min():.2f}")
            with col3:
                st.metric("Maximum", f"{data[selected_column].max():.2f}")
            with col4:
                st.metric("Odchylenie std.", f"{data[selected_column].std():.2f}")
        
        elif analysis_type == "Korelacje":
            st.subheader("Analiza korelacji")
            if 'data' in st.session_state:
                corr_matrix = create_correlation_matrix(st.session_state.data)
                st.plotly_chart(corr_matrix, use_container_width=True)
                
                # Szczegółowa analiza korelacji
                st.subheader("Szczegółowa analiza korelacji")
                correlations = analyze_correlations(st.session_state.data)
                st.write(correlations)
        
        elif analysis_type == "Statystyki opisowe":
            st.subheader("Statystyki opisowe")
            if 'data' in st.session_state:
                stats = calculate_statistics(st.session_state.data)
                st.write(stats)
                
                # Generowanie raportu
                if st.button("Generuj raport statystyczny"):
                    report = generate_statistics_report(st.session_state.data)
                    st.download_button(
                        label="Pobierz raport",
                        data=report,
                        file_name="statistics_report.html",
                        mime="text/html"
                    )
        
        else:  # Analiza rozkładów
            st.subheader("Analiza rozkładów")
            if 'data' in st.session_state:
                selected_column = st.selectbox(
                    "Wybierz kolumnę do analizy",
                    st.session_state.data.select_dtypes(include=[np.number]).columns
                )
                
                # Histogram
                fig = px.histogram(
                    st.session_state.data,
                    x=selected_column,
                    title=f"Rozkład {selected_column}",
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Box plot
                fig = px.box(
                    st.session_state.data,
                    y=selected_column,
                    title=f"Box plot {selected_column}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif train_button or 'training_results' in st.session_state:
        # Sekcja treningu
        st.header("🧠 Trening modeli AI")
        
        if train_button:
            if not models_to_train:
                st.error("❌ Wybierz przynajmniej jeden model do trenowania!")
                st.stop()
            
            # Przygotowanie danych
            with st.spinner("Przygotowywanie danych..."):
                X, y, feature_names = create_sequences(data, st.session_state['params']['seq_length'], target_column)
                
                # Podział na zbiory
                train_size = int(0.8 * len(X))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                st.success("✅ Dane przygotowane pomyślnie")
            
            # Trening modeli
            results = {}
            progress_bar = st.progress(0)
            
            for i, model_name in enumerate(models_to_train):
                with st.spinner(f"Trenowanie modelu {model_name}..."):
                    predictions = simulate_model_training(X_train, y_train, X_test, y_test, model_name, st.session_state['params'])
                    metrics = calculate_metrics(y_test, predictions)
                    
                    results[model_name] = {
                        'predictions': predictions,
                        'actuals': y_test,
                        'metrics': metrics
                    }
                    
                    progress_bar.progress((i + 1) / len(models_to_train))
            
            st.session_state['training_results'] = results
            st.session_state['test_data'] = {'X_test': X_test, 'y_test': y_test}
            
            st.success("✅ Trening zakończony!")
        
        # Wyświetlanie wyników
        if 'training_results' in st.session_state:
            results = st.session_state['training_results']
            
            st.subheader("📈 Wyniki treningu")
            
            # Metryki
            metrics_df = pd.DataFrame({
                model: results[model]['metrics'] 
                for model in results.keys()
            }).T
            
            # Kolorowanie najlepszych wyników
            styled_df = metrics_df.style.highlight_min(subset=['MSE', 'MAE', 'RMSE', 'MAPE (%)', 'Max Error'], color='lightgreen')
            styled_df = styled_df.highlight_max(subset=['R²'], color='lightgreen')
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Najlepszy model
            best_model = min(results.keys(), key=lambda x: results[x]['metrics']['MSE'])
            best_metrics = results[best_model]['metrics']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏆 Najlepszy model", best_model)
            with col2:
                st.metric("📊 R² Score", f"{best_metrics['R²']:.3f}")
            with col3:
                st.metric("📉 MAPE", f"{best_metrics['MAPE (%)']:.1f}%")
            
            # Wykresy predykcji
            st.subheader("📈 Wizualizacja predykcji")
            
            for model_name in results.keys():
                fig = go.Figure()
                
                # Rzeczywiste wartości
                fig.add_trace(go.Scatter(
                    y=results[model_name]['actuals'],
                    name='Rzeczywiste wartości',
                    line=dict(color='blue', width=2)
                ))
                
                # Predykcje
                fig.add_trace(go.Scatter(
                    y=results[model_name]['predictions'],
                    name=f'Predykcje {model_name}',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f"Predykcje vs Rzeczywiste wartości - {model_name}",
                    xaxis_title="Próbka",
                    yaxis_title="Napięcie [V]",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # INTELIGENTNE REKOMENDACJE
            st.subheader("🎯 Inteligentne rekomendacje")
            
            recommendations = generate_recommendations(results, st.session_state['params'])
            
            if recommendations:
                st.markdown("### 💡 Sugerowane ulepszenia:")
                
                for rec in recommendations:
                    with st.expander(f"{rec['category']}: {rec['issue']}"):
                        st.write(f"**Problem:** {rec['issue']}")
                        st.write(f"**Rekomendacja:** {rec['recommendation']}")
                        
                        if rec['action'] != 'none' and rec['action'] != 'add_model':
                            if st.button(f"Zastosuj: {rec['action']} = {rec['new_value']}", key=f"apply_{rec['action']}"):
                                st.session_state['params'][rec['action']] = rec['new_value']
                                st.success(f"✅ Zaktualizowano {rec['action']} na {rec['new_value']}")
                                st.rerun()
                        elif rec['action'] == 'add_model':
                            if st.button(f"Dodaj model {rec['new_value']}", key=f"add_{rec['new_value']}"):
                                if rec['new_value'] not in st.session_state['selected_models']:
                                    st.session_state['selected_models'].append(rec['new_value'])
                                    st.success(f"✅ Dodano model {rec['new_value']} do treningu")
                                    st.rerun()
                
                # Przycisk do ponownego treningu z sugerowanymi parametrami
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Trenuj ponownie z nowymi parametrami", type="primary"):
                        del st.session_state['training_results']
                        st.rerun()
                
                with col2:
                    if st.button("📊 Reset parametrów"):
                        st.session_state['params'] = {
                            'seq_length': 50,
                            'batch_size': 32,
                            'epochs': 30,
                            'learning_rate': 0.001,
                            'hidden_size': 64,
                            'num_layers': 2
                        }
                        st.success("✅ Parametry zresetowane")
                        st.rerun()
            
            else:
                st.success("🎉 Doskonałe wyniki! Model jest gotowy do wdrożenia.")
                
                # Podsumowanie końcowe
                st.subheader("📋 Podsumowanie końcowe")
                
                best_r2 = best_metrics['R²']
                best_mape = best_metrics['MAPE (%)']
                
                if best_r2 > 0.9:
                    st.success(f"✅ **Doskonała jakość predykcji** (R² = {best_r2:.3f})")
                elif best_r2 > 0.8:
                    st.warning(f"⚠️ **Dobra jakość predykcji** (R² = {best_r2:.3f})")
                else:
                    st.error(f"❌ **Niska jakość predykcji** (R² = {best_r2:.3f})")
                
                if best_mape < 5:
                    st.success(f"✅ **Bardzo niski błąd** (MAPE = {best_mape:.1f}%)")
                elif best_mape < 10:
                    st.warning(f"⚠️ **Akceptowalny błąd** (MAPE = {best_mape:.1f}%)")
                else:
                    st.error(f"❌ **Wysoki błąd** (MAPE = {best_mape:.1f}%)")
                
                st.info(f"🏆 **Rekomendowany model:** {best_model}")