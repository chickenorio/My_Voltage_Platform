import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

def train_model(model, train_loader, val_loader, epochs, learning_rate):
    """
    Trenuje model estymacji napięć z wczesnym zatrzymaniem.
    
    Args:
        model: model PyTorch do trenowania
        train_loader: DataLoader z danymi treningowymi
        val_loader: DataLoader z danymi walidacyjnymi
        epochs: liczba epok
        learning_rate: współczynnik uczenia
        
    Returns:
        train_losses, val_losses: listy strat podczas treningu
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    model.train()
    
    for epoch in range(epochs):
        # Faza treningu
        epoch_train_loss = 0.0
        model.train()
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # Dodatkowa strata fizyczna dla PINN
            if hasattr(model, 'physics_loss'):
                physics_loss = model.physics_loss(batch_X, outputs)
                loss += 0.01 * physics_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping dla stabilności
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Faza walidacji
        epoch_val_loss = 0.0
        model.eval()
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                epoch_val_loss += loss.item()
        
        # Obliczenie średnich strat
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
            
        # Progress info co 10 epok
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_model(model, val_loader, scaler_y):
    """
    Ewaluuje model i oblicza metryki.
    
    Args:
        model: wytrenowany model
        val_loader: DataLoader z danymi walidacyjnymi
        scaler_y: scaler do denormalizacji celów
        
    Returns:
        predictions, actuals, metrics: predykcje, rzeczywiste wartości i metryki
    """
    model.eval()
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            
            # Denormalizacja
            predictions = scaler_y.inverse_transform(outputs.squeeze().cpu().numpy().reshape(-1, 1)).flatten()
            actuals = scaler_y.inverse_transform(batch_y.cpu().numpy().reshape(-1, 1)).flatten()
            
            all_predictions.extend(predictions)
            all_actuals.extend(actuals)
    
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    
    # Obliczenie metryk
    mse = mean_squared_error(all_actuals, all_predictions)
    mae = mean_absolute_error(all_actuals, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_actuals, all_predictions)
    
    # Dodatkowe metryki dla sieci elektrycznych
    mape = np.mean(np.abs((all_actuals - all_predictions) / all_actuals)) * 100
    max_error = np.max(np.abs(all_actuals - all_predictions))
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE (%)': mape,
        'Max Error': max_error
    }
    
    return all_predictions, all_actuals, metrics

def create_predictions_plot(actuals, predictions, model_name, target_column):
    """
    Tworzy wykres porównujący predykcje z rzeczywistymi wartościami.
    
    Args:
        actuals: rzeczywiste wartości
        predictions: predykcje modelu
        model_name: nazwa modelu
        target_column: nazwa kolumny docelowej
        
    Returns:
        fig: obiekt Plotly Figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'Predykcje vs Rzeczywiste - {model_name}',
            'Scatter Plot',
            'Błędy predykcji',
            'Histogram błędów'
        ],
        specs=[[{"colspan": 2}, None],
               [{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # Główny wykres czasowy
    sample_indices = np.arange(len(actuals))
    
    fig.add_trace(
        go.Scatter(
            x=sample_indices,
            y=actuals,
            name='Rzeczywiste',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=sample_indices,
            y=predictions,
            name='Predykcje',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=actuals,
            y=predictions,
            mode='markers',
            name='Predykcje vs Rzeczywiste',
            marker=dict(color='green', size=4, opacity=0.6),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Linia idealna (y=x)
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Linia idealna',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Błędy
    errors = predictions - actuals
    fig.add_trace(
        go.Scatter(
            x=sample_indices,
            y=errors,
            mode='lines',
            name='Błędy',
            line=dict(color='orange'),
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Histogram błędów - używamy go.Histogram zamiast add_trace
    fig.add_trace(
        go.Histogram(
            x=errors,
            name='Rozkład błędów',
            nbinsx=30,
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Aktualizacja layoutu
    fig.update_layout(
        height=800,
        title_text=f"Analiza predykcji - {model_name} ({target_column})",
        showlegend=True
    )
    
    # Aktualizacja osi
    fig.update_xaxes(title_text="Próbka", row=1, col=1)
    fig.update_yaxes(title_text=f"{target_column}", row=1, col=1)
    
    fig.update_xaxes(title_text="Rzeczywiste wartości", row=2, col=1)
    fig.update_yaxes(title_text="Predykcje", row=2, col=1)
    
    fig.update_xaxes(title_text="Próbka", row=2, col=2)
    fig.update_yaxes(title_text="Błąd", row=2, col=2)
    
    return fig

def generate_report(results, data, target_column, seq_length, batch_size, epochs, learning_rate):
    """
    Generuje raport HTML z wynikami estymacji napięć.
    
    Args:
        results: słownik z wynikami wszystkich modeli
        data: dane wejściowe
        target_column: kolumna docelowa
        seq_length, batch_size, epochs, learning_rate: parametry treningu
        
    Returns:
        str: HTML content raportu
    """
    
    # Nagłówek raportu
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raport Estymacji Napięć Sieci NN</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .section {{ margin: 30px 0; }}
            .metrics-table {{ border-collapse: collapse; width: 100%; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            .metrics-table th {{ background-color: #f2f2f2; }}
            .best-model {{ background-color: #d4edda; font-weight: bold; }}
            .parameters {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚡ Raport Estymacji Napięć Sieci Niskiego Napięcia</h1>
            <p>Wygenerowano: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Parametry analizy</h2>
            <div class="parameters">
                <p><strong>Kolumna docelowa:</strong> {target_column}</p>
                <p><strong>Długość sekwencji:</strong> {seq_length}</p>
                <p><strong>Batch size:</strong> {batch_size}</p>
                <p><strong>Liczba epok:</strong> {epochs}</p>
                <p><strong>Learning rate:</strong> {learning_rate}</p>
                <p><strong>Liczba rekordów:</strong> {len(data):,}</p>
                <p><strong>Zakres danych:</strong> {data['time'].iloc[0]} - {data['time'].iloc[-1]}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>🏆 Wyniki modeli</h2>
            <table class="metrics-table">
                <tr>
                    <th>Model</th>
                    <th>MSE</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>R²</th>
                    <th>MAPE (%)</th>
                    <th>Max Error</th>
                </tr>
    """
    
    # Znajdź najlepszy model
    best_model = min(results.keys(), key=lambda x: results[x]['metrics']['MSE'])
    
    # Dodaj wiersze z metrykami
    for model_name in results.keys():
        metrics = results[model_name]['metrics']
        row_class = 'best-model' if model_name == best_model else ''
        
        html_content += f"""
                <tr class="{row_class}">
                    <td>{model_name}</td>
                    <td>{metrics['MSE']:.4f}</td>
                    <td>{metrics['MAE']:.4f}</td>
                    <td>{metrics['RMSE']:.4f}</td>
                    <td>{metrics['R²']:.4f}</td>
                    <td>{metrics['MAPE (%)']:.2f}</td>
                    <td>{metrics['Max Error']:.4f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            <p><em>Najlepszy model wyróżniony kolorem zielonym (najniższe MSE)</em></p>
        </div>
        
        <div class="section">
            <h2>📈 Statystyki danych</h2>
    """
    
    # Dodaj statystyki danych
    numeric_cols = ['current', 'active_power', 'reactive_power', 'frequency', 'irradiance']
    available_cols = [col for col in numeric_cols if col in data.columns]
    
    html_content += '<table class="metrics-table"><tr><th>Parametr</th><th>Średnia</th><th>Odch. std.</th><th>Min</th><th>Max</th></tr>'
    
    for col in available_cols:
        stats = data[col].describe()
        html_content += f"""
            <tr>
                <td>{col}</td>
                <td>{stats['mean']:.3f}</td>
                <td>{stats['std']:.3f}</td>
                <td>{stats['min']:.3f}</td>
                <td>{stats['max']:.3f}</td>
            </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>🔍 Analiza wyników</h2>
            <h3>Interpretacja metryk:</h3>
            <ul>
                <li><strong>MSE (Mean Squared Error):</strong> Średni błąd kwadratowy - niższe wartości są lepsze</li>
                <li><strong>MAE (Mean Absolute Error):</strong> Średni błąd bezwzględny - łatwiejsza interpretacja</li>
                <li><strong>RMSE (Root Mean Squared Error):</strong> Pierwiastek z MSE - w tych samych jednostkach co dane</li>
                <li><strong>R² (Coefficient of Determination):</strong> Współczynnik determinacji - bliżej 1.0 jest lepiej</li>
                <li><strong>MAPE (Mean Absolute Percentage Error):</strong> Średni błąd procentowy bezwzględny</li>
                <li><strong>Max Error:</strong> Maksymalny błąd predykcji</li>
            </ul>
            
            <h3>Rekomendacje:</h3>
            <ul>
    """
    
    # Dodaj rekomendacje na podstawie wyników
    best_r2 = results[best_model]['metrics']['R²']
    best_mape = results[best_model]['metrics']['MAPE (%)']
    
    if best_r2 > 0.9:
        html_content += '<li>✅ Doskonała jakość predykcji (R² > 0.9) - model gotowy do wdrożenia</li>'
    elif best_r2 > 0.8:
        html_content += '<li>⚠️ Dobra jakość predykcji (R² > 0.8) - rozważ dodatkową optymalizację</li>'
    else:
        html_content += '<li>❌ Niska jakość predykcji (R² < 0.8) - wymagane dodatkowe prace nad modelem</li>'
    
    if best_mape < 5:
        html_content += '<li>✅ Niski błąd procentowy (MAPE < 5%) - model bardzo precyzyjny</li>'
    elif best_mape < 10:
        html_content += '<li>⚠️ Umiarkowany błąd procentowy (MAPE < 10%) - akceptowalna precyzja</li>'
    else:
        html_content += '<li>❌ Wysoki błąd procentowy (MAPE > 10%) - wymagana poprawa modelu</li>'
    
    html_content += f"""
                <li>🏆 Najlepszy model: <strong>{best_model}</strong></li>
                <li>📊 Rozważ użycie modelu {best_model} do estymacji napięć w sieci</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>⚙️ Szczegóły techniczne</h2>
            <p><strong>Architektura modeli:</strong></p>
            <ul>
                <li><strong>LSTM:</strong> Long Short-Term Memory - sieć rekurencyjna do szeregów czasowych</li>
                <li><strong>PINN:</strong> Physics-Informed Neural Network - uwzględnia prawa fizyki</li>
                <li><strong>Transformer:</strong> Architektura z mechanizmem uwagi</li>
                <li><strong>Hybrid:</strong> Kombinacja różnych architektur</li>
            </ul>
            
            <p><strong>Zastosowanie:</strong></p>
            <p>Model może być użyty do estymacji napięć w punktach krytycznych sieci niskiego napięcia, 
            wspierając operatora w regulacji napięcia i zapobieganiu przekroczeniom normatywnym.</p>
        </div>
        
        <footer style="margin-top: 50px; text-align: center; color: #666;">
            <p>Wygenerowano przez System Estymacji Napięć Sieci NN</p>
        </footer>
    </body>
    </html>
    """
    
    return html_content

def calculate_physics_constraints(predictions, features):
    """
    Oblicza naruszenia fizycznych ograniczeń w sieciach elektrycznych.
    
    Args:
        predictions: predykcje napięcia
        features: cechy wejściowe (prąd, moc, etc.)
        
    Returns:
        dict: słownik z analizą ograniczeń fizycznych
    """
    constraints = {}
    
    # Sprawdzenie zakresu napięcia (przyjmujemy sieć 230V ±10%)
    voltage_min = 207  # 90% z 230V
    voltage_max = 253  # 110% z 230V
    
    voltage_violations = np.sum((predictions < voltage_min) | (predictions > voltage_max))
    constraints['voltage_violations'] = {
        'count': voltage_violations,
        'percentage': (voltage_violations / len(predictions)) * 100
    }
    
    # Sprawdzenie fizycznych zależności
    if features.shape[1] >= 3:  # current, active_power, reactive_power
        current = features[:, 0]
        active_power = features[:, 1]
        reactive_power = features[:, 2]
        
        # Sprawdzenie relacji moc-prąd-napięcie
        calculated_current = np.sqrt(active_power**2 + reactive_power**2) / predictions
        current_discrepancy = np.abs(current - calculated_current)
        
        constraints['power_current_consistency'] = {
            'mean_discrepancy': np.mean(current_discrepancy),
            'max_discrepancy': np.max(current_discrepancy)
        }
    
    return constraints