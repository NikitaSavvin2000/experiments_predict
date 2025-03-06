import os
import requests
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import webbrowser
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
from config import logger
from pathlib import Path

# Изначальные данные
df_init = pd.read_csv('src/data/load_consumption_2025.csv')
# df_init = df_init.iloc[:100000]

measurement = 'load_consumption'

print(df_init.head())
print(f'Колонки - {df_init.columns}')

home_path = os.getcwd()

url_backend = os.getenv("BACKEND_URL", 'http://77.37.136.11:7070')

# Метод который из даты делает вектор (Time2Vec)
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


# Метод который восстанавливает данные из вектора
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

json_list_general_norm_df = df_init.to_dict(orient='records')

logger.info("Normalizing the data.")
df_general_norm_df, min_val, max_val = normalization_request(
    col_time='datetime',
    col_target=measurement,
    json_list_df=json_list_general_norm_df
)

print(df_general_norm_df.head())
print(f'Колонки после нормализации - {df_general_norm_df.columns}')


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length].reshape(self.seq_length, 1)
        y = self.data[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


data = df_general_norm_df[measurement].values

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
train_data, val_data = train_test_split(train_data, test_size=0.2, shuffle=False)
print(test_data)


seq_length = 24
train_dataset = TimeSeriesDataset(train_data, seq_length)
val_dataset = TimeSeriesDataset(val_data, seq_length)
test_dataset = TimeSeriesDataset(test_data, seq_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, nhead=8, num_layers=6):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = x.squeeze(-1)
        x = self.embedding(x.unsqueeze(-1))
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        return x

input_dim = 1
d_model = 16
output_dim = 1
nhead = 4
num_layers = 6

batch_size = 32

# model = TransformerModel(input_dim, output_dim, nhead)
model = TransformerModel(input_dim, d_model, output_dim, nhead, num_layers)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Переменные для отслеживания лучшей модели
best_val_loss = float('inf')
best_epoch = 0
model_save_path = 'best_model.pth'

num_epochs = 10

for batch_x, batch_y in train_loader:
    print("Batch shape before model:", batch_x.shape)
    output = model(batch_x)
    print("Output shape:", output.shape)
    break

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

model.load_state_dict(torch.load(model_save_path))
model.eval()


predictions = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        output = model(batch_x)
        predictions.append(output.numpy())


predictions = np.concatenate(predictions).flatten()

print(f'predictions = {predictions}')
print()

# Обратная нормализация
df_predict_norm = pd.DataFrame({
    measurement: predictions.flatten(), 
    'datetime': df_general_norm_df['datetime'].values[-len(predictions):] 
})

json_list_df_predict_norm = df_general_norm_df.to_dict(orient='records')

df_predict_denorm = reverse_normalization_request(
    col_time='datetime',
    col_target=measurement,
    json_list_norm_df=json_list_df_predict_norm,
    min_val=min_val,
    max_val=max_val
)

print(df_predict_denorm.head())

# Оценка модели
y_test = test_data[seq_length:]
y_pred = predictions

print(predictions)

mape = 100 * mean_absolute_error(y_test, y_pred) / y_test.mean()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'MAPE: {mape:.2f}%')
print(f'RMSE: {rmse:.2f}')

df = pd.DataFrame({"Index": range(len(predictions)), "Prediction": predictions})

fig = px.line(df, x="Index", y="Prediction", markers=False, title="Predictions Over Time")
fig.show()

if df_predict_denorm is not None and not df_predict_denorm.empty:
    # Получаем реальные тестовые данные в исходном масштабе
    y_test_denorm = test_data[seq_length:] * (max_val - min_val) + min_val

    # Формируем временные метки
    datetime_values = df_general_norm_df['datetime'].values[-len(y_test_denorm):]

    test_start_idx = len(train_data)
    test_end_idx = test_start_idx + len(test_data)
    test_datetime = df_general_norm_df['datetime'].values[test_start_idx:test_end_idx]
    datetime_values = test_datetime[seq_length:]

    # Создаем DataFrame для визуализации
    df_plot = pd.DataFrame({
        'Время': datetime_values,
        'Реальные значения': y_test_denorm,
        'Предсказания модели': df_predict_denorm[measurement].values[:len(datetime_values)]
    }).dropna()

    # Проверка совпадения длин массивов
    assert len(df_plot['Время']) == len(df_plot['Реальные значения']) == len(df_plot['Предсказания модели']), \
        f"Несовпадение длин: Время {len(datetime_values)}, Реальные {len(y_test_denorm)}, Предсказания {len(df_predict_denorm)}"

    # Диагностический вывод
    print("Данные для визуализации:")
    print(df_plot.head())
    print(f"Длина временных меток: {len(datetime_values)}")
    print(f"Реальные значения: {len(y_test_denorm)}")
    print(f"Предсказания: {len(df_predict_denorm)}")

    # Упрощенная визуализация
    fig = px.line(df_plot, 
                  x='Время', 
                  y=['Реальные значения', 'Предсказания модели'],
                  title='Сравнение реальных данных и предсказаний модели')
    
    # Визуализация
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

    # Добавляем вертикальную линию раздела
    train_end_idx = len(train_data) + seq_length
    if train_end_idx < len(df_general_norm_df):
        # Получаем временную метку
        vertical_line_date = pd.to_datetime(
            df_general_norm_df['datetime'].values[train_end_idx]
        )
        
        # Добавляем линию с преобразованием в timestamp
        fig.add_vline(
            x=vertical_line_date.timestamp(),
            line_dash="dot", 
            line_color="gray",
            annotation_text="Начало предсказаний",
            annotation_position="top right"
        )

    fig.show()
    html_file = "prediction_comparison.html"
    fig.write_html(html_file)
    print("График сохранен в prediction_comparison.html")
    webbrowser.open(f"file://{Path(html_file).absolute()}")
else:
    logger.error("Ошибка: Не удалось получить денормализованные предсказания")