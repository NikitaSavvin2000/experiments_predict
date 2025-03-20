# ===============================
# Импорт библиотек
# ===============================
import os
import requests
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import webbrowser
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
from config import logger
from pathlib import Path

# ===============================
# Загрузка данных и начальная настройка
# ===============================
df_init = pd.read_csv('src/data/load_consumption_2025.csv')
df_init['datetime'] = pd.to_datetime(df_init['datetime'])
df_init.set_index('datetime', inplace=True)
measurement = 'load_consumption'
print(df_init.head())
print(f'Колонки - {df_init.columns}')

home_path = os.getcwd()
url_backend = os.getenv("BACKEND_URL", 'http://77.37.136.11:7070')

def add_time_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

df_init = add_time_features(df_init)

# ===============================
# Функции для нормализации и обратной нормализации данных
# ===============================
def normalization_request(col_time, col_target, json_list_df):
    url = f'{url_backend}/backend/v1/normalization'
    json = {"col_time": col_time, "col_target": col_target, "json_list_df": json_list_df}
    try:
        req = requests.post(url=url, json=json)
        if req.status_code == 200:
            response_json = req.json()
            norm_df = pd.DataFrame.from_dict(response_json['df_all_data_norm'])
            min_val = float(response_json['min_val'])
            max_val = float(response_json['max_val'])
            return norm_df, min_val, max_val
        else:
            logger.error(f'Status code backend server: {req.status_code}')
            return None, None, None
    except Exception as e:
        logger.error(e)
        return None, None, None

def reverse_normalization_request(col_time, col_target, json_list_norm_df, min_val, max_val):
    url = f'{url_backend}/backend/v1/reverse_normalization'
    json = {
        "col_time": col_time,
        "col_target": col_target,
        "min_val": min_val,
        "max_val": max_val,
        "json_list_norm_df": json_list_norm_df
    }
    try:
        req = requests.post(url=url, json=json)
        if req.status_code == 200:
            reverse_de_norm_data_json = req.json()
            reverse_norm_df = pd.DataFrame.from_dict(reverse_de_norm_data_json['df_all_data_reverse_norm'])
            return reverse_norm_df
        else:
            logger.error(f'Status code backend server: {req.status_code}')
            return None
    except Exception as e:
        logger.error(e)

# ===============================
# Нормализация данных
# ===============================
# Преобразование datetime в строку перед сериализацией
df_init_reset = df_init.reset_index()
df_init_reset['datetime'] = df_init_reset['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Преобразуем DataFrame в список словарей
json_list_general_norm_df = df_init_reset.to_dict(orient='records')

logger.info("Нормализация данных.")
# Проверка формата данных перед отправкой на сервер
print("Пример данных для нормализации:", json_list_general_norm_df[:3])
print("Ключи в данных:", json_list_general_norm_df[0].keys())
print("Тип данных datetime:", type(json_list_general_norm_df[0]['datetime']))
print("Тип данных load_consumption:", type(json_list_general_norm_df[0]['load_consumption']))
print("Пример datetime:", json_list_general_norm_df[0]['datetime'])

# Отправка данных на сервер для нормализации
df_general_norm_df, min_val, max_val = normalization_request(
    col_time='datetime',
    col_target=measurement,
    json_list_df=json_list_general_norm_df
)

if df_general_norm_df is None:
    logger.error("Ошибка: Не удалось нормализовать данные.")
else:
    # Удаление столбца datetime из нормализованных данных
    df_general_norm_df = df_general_norm_df.drop(columns=['datetime'])
    print(df_general_norm_df.head())
    print(f'Колонки после нормализации - {df_general_norm_df.columns}')

# ===============================
# Подготовка данных для обучения модели
# ===============================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx + self.seq_length].values
        y = self.data.iloc[idx + self.seq_length][measurement]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Выбор признаков для обучения
col_for_train = [
    measurement, "hour", "day_of_week", "month", "day_of_year", "is_weekend"
]

# Фильтрация данных
df = df_general_norm_df[col_for_train]

# Разделение данных на обучающую, валидационную и тестовую выборки
train_index = int(len(df) * 0.8)
val_index = int(len(df) * 0.9)

train_data = df.iloc[:train_index]
val_data = df.iloc[train_index:val_index]
test_data = df.iloc[val_index:]

seq_length = 24
train_dataset = TimeSeriesDataset(train_data, seq_length)
val_dataset = TimeSeriesDataset(val_data, seq_length)
test_dataset = TimeSeriesDataset(test_data, seq_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ===============================
# Определение архитектуры модели
# ===============================
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, nhead=8, num_layers=6):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])  
        return x

input_dim = len(col_for_train)  
d_model = 16
output_dim = 1
nhead = 4
num_layers = 6
batch_size = 32

model = TransformerModel(input_dim, d_model, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ===============================
# Обучение модели
# ===============================
best_val_loss = float('inf')
best_epoch = 0
model_save_path = 'best_model.pth'
num_epochs = 10
patience = 5
trigger_times = 0

for epoch in range(num_epochs):
    # Тренировочный этап
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Валидационный этап
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Сохранение лучшей модели
    if val_loss < best_val_loss:
        trigger_times = 0
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_save_path)
        print(f"Лучшая модель сохранена (Эпоха {best_epoch}, Val Loss: {best_val_loss:.4f})")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Ранняя остановка на эпохе {epoch+1}")
            break

# ===============================
# Тестирование модели
# ===============================
model.load_state_dict(torch.load(model_save_path))
model.eval()
predictions = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        output = model(batch_x)
        predictions.append(output.numpy())

predictions = np.concatenate(predictions).flatten()
print(f'predictions = {predictions}')

# ===============================
# Обратная нормализация и оценка модели
# ===============================
df_predict_norm = pd.DataFrame({
    measurement: predictions.flatten(),
    'datetime': df_init_reset['datetime'].values[-len(predictions):]  # Берем временные метки из исходного датасета
})

json_list_df_predict_norm = df_predict_norm.to_dict(orient='records')
df_predict_denorm = reverse_normalization_request(
    col_time='datetime',
    col_target=measurement,
    json_list_norm_df=json_list_df_predict_norm,
    min_val=min_val,
    max_val=max_val
)

print(df_predict_denorm.head())

y_test = test_data[seq_length:]
y_pred = predictions
mape = 100 * mean_absolute_error(y_test, y_pred) / y_test.mean()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'MAPE: {mape:.2f}%')
print(f'RMSE: {rmse:.2f}')

# ===============================
# Визуализация исходных данных
# ===============================
df_init[measurement].plot(title='Исходные данные', figsize=(12, 6))
plt.xlabel('Индекс')
plt.ylabel('Потребление')
plt.grid(True)
plt.show()

# ===============================
# Визуализация результатов
# ===============================
df = pd.DataFrame({"Index": range(len(predictions)), "Prediction": predictions})
fig = px.line(df, x="Index", y="Prediction", markers=False, title="Predictions Over Time")
fig.show()

if df_predict_denorm is not None and not df_predict_denorm.empty:
    y_test_denorm = test_data[seq_length:] * (max_val - min_val) + min_val
    datetime_values = df_init_reset['datetime'].values[-len(y_test_denorm):]

    df_plot = pd.DataFrame({
        'Время': datetime_values,
        'Реальные значения': y_test,
        'Предсказания модели': predictions
    })

    assert len(df_plot['Время']) == len(df_plot['Реальные значения']) == len(df_plot['Предсказания модели']), \
        f"Несовпадение длин: Время {len(datetime_values)}, Реальные {len(y_test_denorm)}, Предсказания {len(df_predict_denorm)}"

    fig = px.line(df_plot, 
                  x='Время', 
                  y=['Реальные значения', 'Предсказания модели'],
                  title='Сравнение реальных данных и предсказаний модели',
                  labels={'value': 'Потребление', 'variable': 'Тип данных'},
                  color_discrete_map={'Реальные значения': '#2E86C1', 'Предсказания модели': '#E74C3C'})

    fig.update_layout(
        xaxis_title='Дата и время',
        yaxis_title='Потребление',
        legend_title='Легенда',
        hovermode='x unified',
        template='plotly_white',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1 час", step="hour", stepmode="backward"),
                    dict(count=6, label="6 часов", step="hour", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    fig.show()
    html_file = "prediction_comparison.html"
    fig.write_html(html_file)
    print("График сохранен в prediction_comparison.html")
    webbrowser.open(f"file://{Path(html_file).absolute()}")
else:
    logger.error("Ошибка: Не удалось получить денормализованные предсказания")