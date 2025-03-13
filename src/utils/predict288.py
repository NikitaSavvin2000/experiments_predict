import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
import webbrowser
from pathlib import Path

# Параметры
seq_length = 24
forecast_steps = 288  # Количество шагов для предсказания

# Класс TransformerModel
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, nhead=4, num_layers=6):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x.unsqueeze(-1))
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])

# Функция для пошагового предсказания
def forecast(model, last_seq, steps=288):
    model.eval()
    forecasted = []
    current_seq = last_seq.copy()

    with torch.no_grad():
        for i in range(steps):
            current_seq_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
            next_val = model(current_seq_tensor).item()
            forecasted.append(next_val)
            current_seq = np.append(current_seq[1:], next_val)

            # Вывод для диагностики
            if i % 50 == 0 or i == steps - 1:
                print(f'Step {i + 1}/{steps} - Next Value: {next_val:.4f}')

    return np.array(forecasted)

# Подготовка данных (пример данных)
df_general_norm_df = pd.read_csv('src/data/load_consumption_2025.csv')
full_data = df_general_norm_df['load_consumption'].values

# Создание и настройка модели
input_dim = 1
model = TransformerModel(input_dim, d_model=16, output_dim=1, nhead=4, num_layers=6)
model.load_state_dict(torch.load('best_model.pth'))  # Загрузка обученной модели

# Получение последней последовательности данных
last_seq = full_data[-seq_length:]

# Прогноз на 288 шагов
print("Начало предсказания на 288 шагов...")
predictions_288 = forecast(model, last_seq, steps=forecast_steps)
print("Предсказание завершено.")

# Визуализация
fig = px.line(pd.DataFrame({"Index": range(len(predictions_288)), "Prediction": predictions_288}),
              x="Index", y="Prediction", title="288 Steps Forecast")
fig.show()

html_file = "prediction288_comparison.html"
fig.write_html(html_file)
print("График сохранен в prediction288_comparison.html")
webbrowser.open(f"file://{Path(html_file).absolute()}")
