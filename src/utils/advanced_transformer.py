import os
import torch
import requests
import psycopg2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
from tqdm import tqdm
from config import logger
import random
from plotly.subplots import make_subplots
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import webbrowser
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Установка сидов для воспроизводимости
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Для GPU
    torch.backends.cudnn.deterministic = True  # Для детерминированного поведения
    torch.backends.cudnn.benchmark = False     # Отключение автотюнинга
set_seed(42)

# Конфигурация
home_path = os.getcwd()
path_to_save = f'{home_path}/src/models_res'
LAG = 5
HORIZON = 288
BATCH_SIZE = 1
EPOCHS = 10
LR = 0.1
D_MODEL = 2
NHEAD = 2
NUM_LAYERS = 2
DROPOUT = 0.2
points_per_call = LAG * 4
measurement = 'load_consumption'
url_backend = os.getenv("BACKEND_URL", 'http://77.37.136.11:7070')

# Колонки для обучения
col_for_train = [measurement, "year", "month", "day", "week", "day_of_week", "hour", "minute", "hour_sin", "hour_cos",
                 "day_of_week_sin", "day_of_week_cos", "week_sin", "week_cos", "month_sin", "month_cos", "part_of_day",
                 "is_night", "is_weekend", "day_of_year"]

# Создание папки для сохранения моделей
models_dir = f"{home_path}/src/models_res"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Функции для нормализации и обратной нормализации
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

# Функция для логирования
def cast_logger(message):
    count = len(message) + 4
    if count > 150:
        count = 150
    print('=' * count)
    print(f'>>> {message}')
    print('=' * count)

# Функция для предсказаний
def make_predictions(x_input, x_future, points_per_call, model, device="cpu"):
    model.eval()
    predict_values = []
    x_future_len = len(x_future)
    remaining_horizon = x_future_len
    while remaining_horizon > 0:
        current_points_to_predict = min(remaining_horizon, points_per_call)
        x_input_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
        x_input_tensor = x_input_tensor.unsqueeze(0)
        with torch.no_grad():
            y_predict = model(x_input_tensor)
        y_predict = y_predict.cpu().numpy().flatten()
        y_predict = y_predict[:current_points_to_predict]
        predict_values.extend(y_predict)
        for i in range(current_points_to_predict):
            cur_val = y_predict[i]
            x_input = np.delete(x_input, 0, axis=0)
            future_lag = x_future[0]
            x_future = np.delete(x_future, 0, axis=0)
            future_lag[0] = cur_val
            x_input = np.append(x_input, future_lag.reshape(1, -1), axis=0)
        remaining_horizon -= current_points_to_predict
    return predict_values

# Функция для создания входных данных
def create_x_input(df_train, n_steps):
    df_input = df_train.iloc[len(df_train) - n_steps:]
    x_input = df_input.values
    return x_input

# Функция для загрузки данных из базы данных
def fetch_data_from_db():
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
    SELECT * FROM {table_name} ORDER BY datetime;
    """
    cur.execute(select_query)
    rows = cur.fetchall()
    df_result = pd.DataFrame(rows, columns=["datetime", measurement])
    df_result["datetime"] = df_result["datetime"].dt.tz_localize(None)
    cur.close()
    conn.close()
    return df_result

# Функция для расчета MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Функция для разделения последовательности
def split_sequence(sequence, n_steps, points_per_call):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        out_end_ix = end_ix + points_per_call
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_end_ix, 0]  # Используем только load_consumption
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Класс для работы с временными рядами
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Класс для пулинга с вниманием
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x), dim=1)
        return (x * attn_weights).sum(dim=1)

# Класс модели Transformer
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT, output_seq_len=points_per_call):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, norm_first=False)  # norm_first=False для устранения предупреждения
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.attn_pool = AttentionPooling(d_model)
        self.fc = nn.Linear(d_model, output_seq_len)
    def forward(self, x):
        x = self.embedding(x)  # x: (batch_size, seq_length, d_model)
        x = self.dropout(x)
        x_residual = x
        x = self.layer_norm(x)
        x = self.transformer_encoder(x)  # x: (batch_size, seq_length, d_model)
        x = x + x_residual
        x = self.attn_pool(x)  # x: (batch_size, d_model)
        return self.fc(x)  # x: (batch_size, output_dim)

# Загрузка данных
df_init = fetch_data_from_db()
df_init['datetime'] = df_init['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Нормализация данных
json_list_general_norm_df = df_init.to_dict(orient='records')
logger.info("Normalizing the data.")
df_general_norm_df, min_val, max_val = normalization_request(
    col_time='datetime',
    col_target=measurement,
    json_list_df=json_list_general_norm_df
)
df_general_norm_df = df_general_norm_df.drop(columns=['datetime'])
all_col = df_general_norm_df.columns

# Подготовка данных для обучения
df = df_general_norm_df
diff_cols = all_col.difference(col_for_train)
train_index = int(len(df) - HORIZON)
df_train_all_col = df.iloc[:train_index]
df_test_all_col = df.iloc[train_index:]
df_true_all_col = df_test_all_col.copy()
df = df_general_norm_df[col_for_train]
df_train = df.iloc[:train_index]
values = df_train[col_for_train].values
X, y = split_sequence(values, LAG, points_per_call)
df_test = df.iloc[train_index:]
df_for_comparison = df_init.iloc[train_index:]
df_true = df_test.copy()
df_forecast = df_test.copy()
x_input = create_x_input(df_train, LAG)
df_test = df_test.copy()
x_future = df_test.values
n_features = values.shape[1]
X_tensor = torch.tensor(X, dtype=torch.float32)  
y_tensor = torch.tensor(y, dtype=torch.float32).squeeze(-1) 
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Инициализация модели и оптимизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(input_dim=X.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()

# TensorBoard Logging
writer = SummaryWriter()

# Ранняя остановка и сохранение лучшей модели
patience = 5
best_val_loss = float('inf')
trigger_times = 0

# Обучение модели
progress_bar_epochs = tqdm(range(EPOCHS), desc="Epoch")
for epoch in progress_bar_epochs:
    model.train()
    train_loss = 0.0
    val_loss = 0.0

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        # Mixed Precision Training
        with torch.cuda.amp.autocast():
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        # Логирование метрик после каждых 100 батчей
        if batch_idx % 100 == 0:
            print(f"Эпоха {epoch}, Батч {batch_idx}, Loss: {loss.item():.4f}")
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

    # Валидация
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in train_loader:  # Используем train_loader для валидации
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            val_outputs = model(batch_x)
            val_loss += criterion(val_outputs, batch_y).item()

    val_loss /= len(train_loader)
    print(f"Эпоха {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

    # Логирование в TensorBoard
    writer.add_scalar('Loss/val', val_loss, epoch)

    # Ранняя остановка и сохранение лучшей модели
    if val_loss < best_val_loss:
        trigger_times = 0
        best_val_loss = val_loss
        best_epoch = epoch + 1

        # Сохранение контрольных точек каждые 5 эпох
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(models_dir, f"checkpoint_epoch_{epoch}_val_loss_{val_loss:.4f}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Контрольная точка сохранена (Эпоха {epoch}, Val Loss: {val_loss:.4f})")

        model_filename = f"model_epoch_{epoch + 1}_val_loss_{best_val_loss:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        model_save_path = os.path.join(models_dir, model_filename)
        torch.save(model.state_dict(), model_save_path)
        print(f"Лучшая модель сохранена (Эпоха {best_epoch}, Val Loss: {best_val_loss:.4f})")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Ранняя остановка на эпохе {epoch + 1}")
            break

# Загрузка лучшей модели для тестирования
best_model_path = os.path.join(models_dir, f"model_epoch_{best_epoch}_val_loss_{best_val_loss:.4f}_*.pth")
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Тестирование модели
predictions = []
with torch.no_grad():
    for batch_x, _ in train_loader:  # Используем train_loader для тестирования
        output = model(batch_x)
        predictions.append(output.numpy())
predictions = np.concatenate(predictions).flatten()

# Обратная нормализация
df_predict_norm = pd.DataFrame({
    measurement: predictions.flatten(),
    'datetime': df_init['datetime'].values[-len(predictions):]
})
json_list_df_predict_norm = df_predict_norm.to_dict(orient='records')
df_predict_denorm = reverse_normalization_request(
    col_time='datetime',
    col_target=measurement,
    json_list_norm_df=json_list_df_predict_norm,
    min_val=min_val,
    max_val=max_val
)

# Оценка модели
y_test = df_true[measurement].values[-len(predictions):]
y_pred = predictions
mape = 100 * mean_absolute_error(y_test, y_pred) / y_test.mean()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'MAPE: {mape:.2f}%')
print(f'RMSE: {rmse:.2f}')

# Визуализация результатов
df_plot = pd.DataFrame({
    'Время': df_init['datetime'].values[-len(y_test):],
    'Реальные значения': y_test,
    'Предсказания модели': predictions
})
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
html_file = "prediction_comparison.html"
fig.write_html(html_file)
print("График сохранен в prediction_comparison.html")
webbrowser.open(f"file://{Path(html_file).absolute()}")

# Закрытие TensorBoard
writer.close()