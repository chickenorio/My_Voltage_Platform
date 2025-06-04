import torch
import torch.nn as nn
import numpy as np

class LSTMEstimator(nn.Module):
    """
    Model LSTM do estymacji szeregów czasowych w sieciach niskiego napięcia.
    Idealny do przechwytywania długoterminowych zależności w danych.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMEstimator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dodatkowe warstwy dla lepszej reprezentacji
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        # Inicjalizacja stanów ukrytych
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass przez LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Bierzemy ostatni output z sekwencji
        last_output = lstm_out[:, -1, :]
        
        # Forward pass przez warstwy fully connected
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class PINNEstimator(nn.Module):
    """
    Physics-Informed Neural Network dla estymacji napięć.
    Uwzględnia fizyczne prawa rządzące sieciami elektrycznymi.
    """
    def __init__(self, layers, physics_weight=0.01):
        super(PINNEstimator, self).__init__()
        self.physics_weight = physics_weight
        
        # Budowa sieci neuronowej
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            
        # Funkcje aktywacji
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Bierzemy ostatni krok czasowy z sekwencji
        if len(x.shape) == 3:
            x = x[:, -1, :]  # [batch, seq_len, features] -> [batch, features]
            
        # Forward pass przez sieć
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            
        # Ostatnia warstwa bez aktywacji
        x = self.layers[-1](x)
        return x
    
    def physics_loss(self, x, predictions):
        """
        Implementacja fizycznych ograniczeń dla sieci elektrycznych.
        Bazuje na prawach Kirchhoffa i charakterystykach sieci nn.
        """
        if len(x.shape) == 3:
            x_last = x[:, -1, :]
        else:
            x_last = x
            
        # Ekstraktuj komponenty (zakładając standardowy format danych)
        current = x_last[:, 0:1] if x_last.shape[1] > 0 else torch.zeros_like(predictions)
        active_power = x_last[:, 1:2] if x_last.shape[1] > 1 else torch.zeros_like(predictions)
        reactive_power = x_last[:, 2:3] if x_last.shape[1] > 2 else torch.zeros_like(predictions)
        
        # Prawo Ohma: V = I * R (uproszczone)
        # Związek między mocą a napięciem: P = V * I * cos(φ)
        resistance_assumption = 0.1  # Założona rezystancja
        voltage_from_ohm = current * resistance_assumption
        
        # Fizyczne ograniczenie: napięcie powinno być związane z mocą
        power_constraint = torch.abs(predictions - voltage_from_ohm)
        
        # Ograniczenie ciągłości (napięcie nie może zmieniać się drastycznie)
        continuity_loss = torch.mean(power_constraint ** 2)
        
        # Ograniczenie zakresu napięcia (np. 0.9-1.1 p.u.)
        voltage_range_loss = torch.mean(
            torch.relu(predictions - 1.1) + torch.relu(0.9 - predictions)
        )
        
        return continuity_loss + voltage_range_loss

class PositionalEncoding(nn.Module):
    """Kodowanie pozycyjne dla modelu Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEstimator(nn.Module):
    """
    Model Transformer z mechanizmem uwagi dla estymacji szeregów czasowych.
    Doskonały do przechwytywania długodystansowych zależności.
    """
    def __init__(self, input_size, d_model, nhead, num_layers, 
                 dim_feedforward=256, dropout=0.1):
        super(TransformerEstimator, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Warstwy wyjściowe
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 1)
        
    def forward(self, x):
        # Projekcja wejścia do d_model wymiarów
        x = self.input_projection(x) * np.sqrt(self.d_model)
        
        # Dodanie kodowania pozycyjnego
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]
        
        # Forward pass przez transformer
        transformer_out = self.transformer_encoder(x)
        
        # Bierzemy ostatni output z sekwencji
        last_output = transformer_out[:, -1, :]
        
        # Warstwy klasyfikacyjne
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class HybridEstimator(nn.Module):
    """
    Model hybrydowy łączący różne architektury.
    Wykorzystuje ensemble learning do lepszych predykcji.
    """
    def __init__(self, models, weights=None):
        super(HybridEstimator, self).__init__()
        
        self.models = nn.ModuleDict(models)
        
        # Wagi dla różnych modeli
        if weights is None:
            self.weights = {name: 1.0 / len(models) for name in models.keys()}
        else:
            self.weights = weights
            
        # Dodatkowa warstwa do kombinowania predykcji
        self.combiner = nn.Sequential(
            nn.Linear(len(models), len(models) // 2 + 1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(len(models) // 2 + 1, 1)
        )
        
    def forward(self, x):
        # Zbierz predykcje z wszystkich modeli
        predictions = []
        
        for name, model in self.models.items():
            pred = model(x)
            predictions.append(pred)
            
        # Stack predykcji
        stacked_preds = torch.cat(predictions, dim=1)
        
        # Kombinuj przez learned weights
        combined = self.combiner(stacked_preds)
        
        return combined
    
    def get_individual_predictions(self, x):
        """Zwraca predykcje z poszczególnych modeli"""
        individual_preds = {}
        
        for name, model in self.models.items():
            with torch.no_grad():
                pred = model(x)
                individual_preds[name] = pred.cpu().numpy()
                
        return individual_preds

class EnsembleEstimator(nn.Module):
    """
    Zaawansowany model ensemble z adaptacyjnymi wagami.
    """
    def __init__(self, models, meta_learner_hidden=64):
        super(EnsembleEstimator, self).__init__()
        
        self.models = nn.ModuleDict(models)
        
        # Meta-learner do dynamicznego ważenia predykcji
        self.meta_learner = nn.Sequential(
            nn.Linear(len(models), meta_learner_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(meta_learner_hidden, len(models)),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Zbierz predykcje z wszystkich modeli
        predictions = []
        
        for name, model in self.models.items():
            pred = model(x)
            predictions.append(pred)
            
        # Stack predykcji
        stacked_preds = torch.cat(predictions, dim=1)
        
        # Oblicz adaptacyjne wagi
        weights = self.meta_learner(stacked_preds.detach())
        
        # Ważona suma predykcji
        weighted_pred = torch.sum(stacked_preds * weights, dim=1, keepdim=True)
        
        return weighted_pred