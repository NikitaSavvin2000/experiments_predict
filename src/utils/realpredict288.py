import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import webbrowser
from pathlib import Path

# Параметры
seq_length = 24
forecast_steps = 288  

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

# Функция для получения данных из базы
def fetch_data_from_db(limit=500):  
    table_name = 'load_consumption'
    measurement = 'load_consumption'

    DB_PARAMS = {
        "dbname": "mydb",
        "user": "myuser",
        "password": "mypassword",
        "host": "77.37.136.11",
        "port": 8083
    }

    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    select_query = f"""
    SELECT * FROM {table_name} 
    ORDER BY datetime DESC 
    LIMIT {limit};
    """

    cur.execute(select_query)
    rows = cur.fetchall()

    df_result = pd.DataFrame(rows, columns=["datetime", measurement])
    df_result["datetime"] = pd.to_datetime(df_result["datetime"]).dt.tz_localize(None)

    cur.close()
    conn.close()

    # Поскольку данные загружены в обратном порядке (из-за DESC), нужно перевернуть DataFrame
    return df_result.iloc[::-1].reset_index(drop=True)

# Функция для пошагового предсказания
def forecast(model, initial_seq, steps=288):
    model.eval()
    forecasted = []
    current_seq = initial_seq.copy()

    with torch.no_grad():
        for i in range(steps):
            current_seq_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
            next_val = model(current_seq_tensor).item()
            forecasted.append(next_val)
            current_seq = np.append(current_seq[1:], next_val)

            if i % 50 == 0 or i == steps - 1:
                print(f'Step {i + 1}/{steps} - Next Value: {next_val:.4f}')

    return np.array(forecasted)

# Загрузка данных
df = fetch_data_from_db(limit=500)

df_last_288 = df[-forecast_steps:]
last_seq = df_last_288['load_consumption'].values[-seq_length:]

# Создание и настройка модели
input_dim = 1
model = TransformerModel(input_dim, d_model=16, output_dim=1, nhead=4, num_layers=6)
model.load_state_dict(torch.load('best_model.pth'))  

# Прогноз на 288 шагов
print("Начало предсказания на 288 шагов...")
predictions_288 = forecast(model, last_seq, steps=forecast_steps)
print("Предсказание завершено.")

# Расчёт метрик
mape = 100 * mean_absolute_error(df_last_288['load_consumption'].values, predictions_288) / np.mean(df_last_288['load_consumption'].values)
rmse = np.sqrt(mean_squared_error(df_last_288['load_consumption'].values, predictions_288))
print(f'MAPE: {mape:.2f}%')
print(f'RMSE: {rmse:.2f}')

# Визуализация
df_last_288['Тип'] = "Реальные"
df_pred_plot = pd.DataFrame({
    "datetime": df_last_288['datetime'].values,
    "load_consumption": predictions_288,
    "Тип": "Прогноз"
})

fig = px.line(pd.concat([df_last_288, df_pred_plot]),
              x='datetime', y='load_consumption', color='Тип',
              title='Прогноз на 288 шагов vs Реальные данные',
              color_discrete_map={"Реальные": "#2E86C1", "Прогноз": "#E74C3C"})

fig.show()
html_file = "db_prediction_comparison.html"
fig.write_html(html_file)
print("График сохранен в db_prediction_comparison.html")
webbrowser.open(f"file://{Path(html_file).absolute()}")
