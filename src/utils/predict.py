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
print(test_data)

seq_length = 24
train_dataset = TimeSeriesDataset(train_data, seq_length)
test_dataset = TimeSeriesDataset(test_data, seq_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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

num_epochs = 10

for batch_x, batch_y in train_loader:
    print("Batch shape before model:", batch_x.shape)
    output = model(batch_x)
    print("Output shape:", output.shape)
    break

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.squeeze(-1)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

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
# df_predict_norm = pd.DataFrame({measurement: np.concatenate(predictions)})

# json_list_df_predict_norm = df_general_norm_df.to_dict(orient='records')
#
# df_predict = reverse_normalization_request(
#     col_time='datetime',
#     col_target=measurement,
#     json_list_norm_df=json_list_df_predict_norm,
#     min_val=min_val,
#     max_val=max_val
# )

# print(df_predict.head())

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