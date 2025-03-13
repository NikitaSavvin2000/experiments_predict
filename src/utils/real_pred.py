import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.express as px
from torch.utils.data import Dataset, DataLoader
import os
import os
import requests
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
from config import logger


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

df_init = fetch_data_from_db()

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

# Перед подачей на эндпоинт нужно все превратить в json
json_list_general_norm_df = df_init.to_dict(orient='records')

logger.info("Normalizing the data.")
df_general_norm_df, min_val, max_val = normalization_request(
    col_time='datetime',
    col_target=measurement,
    json_list_df=json_list_general_norm_df
)

print(df_general_norm_df.head())
print(f'Колонки после нормализации - {df_general_norm_df.columns}')


# Определение датасета
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.float32),
            torch.tensor(self.data[idx + self.seq_length], dtype=torch.float32)
        )

# Определение модели
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x.unsqueeze(-1))
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        return x

# Функция для пошагового предсказания
def forecast(model, last_seq, steps=288):
    model.eval()
    forecasted = []
    current_seq = last_seq

    with torch.no_grad():
        for _ in range(steps):
            current_seq_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
            next_val = model(current_seq_tensor).item()
            forecasted.append(next_val)
            current_seq = np.append(current_seq[1:], next_val)

    return np.array(forecasted)

# Подготовка данных
seq_length = 30
measurement = "load_consumption"
full_data = df_general_norm_df[measurement].values
full_dataset = TimeSeriesDataset(full_data, seq_length)
full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

# Параметры модели
# input_dim = 1
# d_model = 64
# output_dim = 1
# nhead = 2
# num_layers = 2

input_dim = 1
d_model = 16  # Должно быть кратно nhead
output_dim = 1
nhead = 4  # nhead должно делиться на d_model
num_layers = 6

# Создание модели
model = TransformerModel(input_dim, d_model, output_dim, nhead, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in full_loader:
        batch_x = batch_x.squeeze(-1)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Получение последней последовательности данных
last_seq = full_data[-seq_length:]

# Прогноз на 288 шагов
predictions_288 = forecast(model, last_seq, steps=288)

print(f'predictions_288 = {predictions_288}')



# Визуализация
fig = px.line(pd.DataFrame({"Index": range(len(predictions_288)), "Prediction": predictions_288}),
              x="Index", y="Prediction", title="288 Steps Forecast")
fig.show()