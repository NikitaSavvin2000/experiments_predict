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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


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


features = len(df_general_norm_df.columns)

for col in df_general_norm_df.columns:
    try:
        df_general_norm_df[col] = pd.to_numeric(df_general_norm_df[col], errors='coerce')
    except:
        print(f"Ошибка преобразования столбца {col}. Проверьте данные на наличие нечисловых значений.")
        # Здесь можно добавить более сложную обработку ошибок (например, удаление столбца, если он содержит много нечисловых данных)



# Масштабирование данных (очень важно для нейронных сетей)
# scaler = StandardScaler()
# numerical_cols = df_general_norm_df.select_dtypes(include=np.number).columns
# df_general_norm_df[numerical_cols] = scaler.fit_transform(df_general_norm_df[numerical_cols])

# Разделяем на признаки (X) и целевую переменную (y)
X = df_general_norm_df.drop('load_consumption', axis=1).values
y = df_general_norm_df['load_consumption'].values

# Разделяем данные на тренировочный и тестовый наборы
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.01, shuffle=False)


seq_length = 24


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.num_features = X.shape[1]

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.seq_length]
        y = self.y[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


train_dataset = TimeSeriesDataset(train_X, train_y, seq_length)
test_dataset = TimeSeriesDataset(test_X, test_y, seq_length)

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
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        return x


input_dim = len(df_general_norm_df.columns) - 1 # Количество признаков (без целевой переменной)
d_model = 16
output_dim = 1
nhead = 4
num_layers = 6
batch_size = 32

model = TransformerModel(input_dim, d_model, output_dim, nhead, num_layers)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1


for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
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