import pandas as pd
import os

# Ścieżki do plików
DATA_DIR = 'data'
SOURCE_DATA_PATH = os.path.join(DATA_DIR, 'network_voltage_data.csv')

def load_working_data():
    """
    Odczytuje dane z pliku working_data.csv.
    Zwraca DataFrame lub None w przypadku błędu.
    """
    try:
        if not os.path.exists('data/working_data.csv'):
            raise FileNotFoundError("Plik data/working_data.csv nie istnieje")
        return pd.read_csv('data/working_data.csv')
    except Exception as e:
        print(f"Błąd odczytu pliku: {str(e)}")
        return None

def save_working_data(data):
    """
    Zapisuje dane do pliku working_data.csv.
    Zwraca True/False w zależności od sukcesu operacji.
    """
    try:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        data.to_csv('data/working_data.csv', index=False)
        return True
    except Exception as e:
        print(f"Błąd zapisu pliku: {str(e)}")
        return False

def initialize_working_data():
    """
    Inicjalizuje plik working_data.csv z danymi źródłowymi.
    Zwraca True/False w zależności od sukcesu operacji.
    """
    try:
        if not os.path.exists(SOURCE_DATA_PATH):
            raise FileNotFoundError(f"Plik źródłowy {SOURCE_DATA_PATH} nie istnieje")
        
        # Sprawdź czy working_data.csv już istnieje
        if os.path.exists('data/working_data.csv'):
            print("Plik data/working_data.csv już istnieje. Nie nadpisuję go.")
            return True
            
        data = pd.read_csv(SOURCE_DATA_PATH)
        return save_working_data(data)
    except Exception as e:
        print(f"Błąd inicjalizacji danych: {str(e)}")
        return False 