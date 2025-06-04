import pandas as pd
import os

# Ścieżki do plików
source_path = os.path.join('data', 'network_voltage_data.csv')
target_path = os.path.join('data', 'working_data.csv')

print("Wczytywanie danych źródłowych...")
data = pd.read_csv(source_path)

print("Zapisywanie do pliku roboczego...")
data.to_csv(target_path, index=False)

print("Sprawdzanie zapisanego pliku...")
saved_data = pd.read_csv(target_path)
print(f"Plik został zapisany pomyślnie. Liczba wierszy: {len(saved_data)}") 