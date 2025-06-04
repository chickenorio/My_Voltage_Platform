# Estymator Stanu Sieci Niskiego Napięcia

System sztucznej inteligencji do estymacji skrajnych napięć w sieciach niskiego napięcia.

## 🎯 Cel projektu

System został opracowany w oparciu o najnowsze badania w dziedzinie estymacji napięć w sieciach niskiego napięcia z wykorzystaniem sztucznej inteligencji. Głównym celem jest estymacja skrajnych napięć w punktach krytycznych sieci nn bazując na pomiarach z stacji SN/nn oraz danych generacji fotowoltaicznej.

## 📁 Struktura projektu

```
my_voltage_platform/
├── data/                  # Dane i logi
│   ├── logs/             # Logi aplikacji
│   └── working_data.csv  # Aktualne dane robocze
├── src/                   # Kod źródłowy
│   ├── anomaly_detector/ # Detekcja anomalii
│   ├── data_ingest/      # Wczytywanie danych
│   ├── missing_handler/  # Obsługa braków
│   ├── visualization/    # Wizualizacje
│   └── correlation/      # Analiza korelacji
├── ui/                    # Interfejsy użytkownika
│   ├── streamlit_app.py  # Aplikacja Streamlit
│   └── gradio_interface.py # Interfejs Gradio
├── utils/                 # Narzędzia pomocnicze
├── tests/                 # Testy
├── requirements.txt       # Zależności
└── README.md             # Dokumentacja
```

## 🚀 Instalacja

1. Sklonuj repozytorium:
```bash
git clone https://github.com/your-username/my_voltage_platform.git
cd my_voltage_platform
```

2. Utwórz i aktywuj środowisko wirtualne:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

## 💻 Uruchomienie

### Aplikacja Streamlit
```bash
streamlit run ui/streamlit_app.py
```

### Interfejs Gradio
```bash
python ui/gradio_interface.py
```

## 📊 Funkcjonalności

- Wizualizacja szeregów czasowych parametrów sieci
- Detekcja i obsługa anomalii
- Wypełnianie brakujących wartości
- Analiza korelacji między parametrami
- Trening i porównanie modeli AI
- Estymacja skrajnych napięć

## 🧠 Modele AI

System wykorzystuje następujące architektury:
- LSTM (Long Short-Term Memory)
- PINN (Physics-Informed Neural Networks)
- Transformer
- Model hybrydowy (Ensemble)

## 📈 Dane wejściowe

System analizuje następujące parametry:
- Prądy fazowe (L1, L2, L3)
- Napięcia fazowe
- Moce czynna i bierna
- Częstotliwość sieci
- Harmoniczne (THD)
- Dane generacji PV
- Dane meteorologiczne
- Parametry topologiczne

## 🤝 Współpraca

1. Fork repozytorium
2. Utwórz branch dla nowej funkcjonalności (`git checkout -b feature/amazing-feature`)
3. Commit zmian (`git commit -m 'Add amazing feature'`)
4. Push do brancha (`git push origin feature/amazing-feature`)
5. Otwórz Pull Request

## 📝 Licencja

Ten projekt jest udostępniany na licencji MIT. Szczegóły w pliku `LICENSE`.

## 📧 Kontakt

Autor: [Twoje Imię] - [email@example.com] 