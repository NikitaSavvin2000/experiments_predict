import os
import torch
import requests
import psycopg2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go

from config import logger
from torch.utils.data import Dataset, DataLoader

home_path = os.getcwd()
path_to_save = f'{home_path}/src/models_res'

LAG = 5
HORIZON = 288
BATCH_SIZE = 64
EPOCHS = 3
LR = 0.0005
D_MODEL = 64
NHEAD = 2
NUM_LAYERS = 2
DROPOUT = 0.1


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


df = fetch_data_from_db()
df_init = df[:-288]
df_test = df[-288:]
print(df_init)
print(df_test)

measurement = 'load_consumption'

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


df_init['datetime'] = df_init['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

json_list_general_norm_df = df_init.to_dict(orient='records')
logger.info("Normalizing the data.")

df_general_norm_df, min_val, max_val = normalization_request(
    col_time='datetime',
    col_target=measurement,
    json_list_df=json_list_general_norm_df
)
print(df_general_norm_df)

df_test['datetime'] = df_test['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

json_list_test_norm_df = df_test.to_dict(orient='records')

logger.info("Normalizing the data.")
df_test_norm, min_val_test, max_val_testv = normalization_request(
    col_time='datetime',
    col_target=measurement,
    json_list_df=json_list_test_norm_df
)


def create_sequences(df, target_col, seq_length):
    features = df.drop(columns=[target_col]).values
    target = df[target_col].values
    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


df = df_general_norm_df

df = df.drop(columns=["datetime"])

X, y = create_sequences(df, "load_consumption", LAG)
X = X.astype(np.float32)
y = y.astype(np.float32)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


class TimeSeriesDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = TimeSeriesDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


class TimeSeriesTransformer(nn.Module):

    def __init__(self, input_dim, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, LAG, d_model))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        return self.fc(x).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(input_dim=X.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss/len(train_loader):.4f}")


model.eval()
future_predictions = []
input_seq = torch.tensor(X[-1], dtype=torch.float32).unsqueeze(0).to(device)
print(f'input_seq = {input_seq}')

save_path = f"{path_to_save}/model_weights.pth"
torch.save(model.state_dict(), save_path)
torch.save(model, f"{path_to_save}/model_full.pth")

with torch.no_grad():
    for _ in range(HORIZON):
        pred = model(input_seq).cpu().item()
        future_predictions.append(pred)
        next_input = np.roll(input_seq.cpu().numpy(), -1, axis=1)
        next_input[0, -1, :-1] = next_input[0, -2, :-1]
        next_input[0, -1, -1] = pred
        input_seq = torch.tensor(next_input, dtype=torch.float32).to(device)

# print(f'future_predictions = {future_predictions}')

real_values = df_test_norm[measurement]

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_value = mean_absolute_percentage_error(real_values, future_predictions)

fig = go.Figure()
fig.add_trace(go.Scatter(y=real_values, mode='lines', name='Real Values', line=dict(color='blue')))
fig.add_trace(go.Scatter(y=future_predictions, mode='lines', name='Predicted Future', line=dict(color='orange')))
fig.update_layout(
    title=f'Actual vs Predicted (MAPE: {mape_value:.2f}%)',
    xaxis_title='Time',
    yaxis_title='Value',
    template='plotly_white'
)

file_path = f"{path_to_save}/file.html"
fig.write_html(file_path)

fig.show()