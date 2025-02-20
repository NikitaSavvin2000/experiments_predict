import os
import requests
import pandas as pd

from config import logger

# Изначальные данные
df_init = pd.read_csv('src/data/load_consumption_2025.csv')

measurement = 'load_consumption'

print(df_init.head())
print(f'Колонки - {df_init.columns}')


home_path = os.getcwd()

url_backend = os.getenv("BACKEND_URL", 'http://77.37.136.11:7070')


# Метод который из даты делает вектор (Time2Vec). Ниже есть пример вызова
def normalization_request(col_time, col_target, json_list_df):

    url = f'{url_backend}/backend/v1/normalization'

    json = {"col_time": col_time, "col_target": col_target, "json_list_df": json_list_df}
    try:
        req = requests.post(
            url=url,
            json=json,
        )
        if req.status_code == 200:
            response_json = req.json()
            norm_df = pd.DataFrame.from_dict(response_json['df_all_data_norm'])
            min_val = float(response_json['min_val'])
            max_val = float(response_json['max_val'])

            return norm_df, min_val, max_val
        else:
            logger.error(f'Status code backend server: {req.status_code}')
            logger.error(req.status_code)
            return None, None, None
    except Exception as e:
        logger.error(e)
        return None, None, None


# Метод который восстанавливает данные из вектора. Ниже есть пример вызова
def reverse_normalization_request(col_time, col_target, json_list_norm_df, min_val, max_val):
    url = f'{url_backend}/backend/v1/reverse_normalization'
    json = {
        "col_time": col_time,
        "col_target": col_target,
        "min_val": min_val,
        "max_val": max_val,
        "json_list_norm_df": json_list_norm_df}
    try:
        req = requests.post(
            url=url,
            json=json,
        )
        if req.status_code == 200:
            reverse_de_norm_data_json = req.json()
            reverse_norm_df = pd.DataFrame.from_dict(reverse_de_norm_data_json['df_all_data_reverse_norm'])
            return reverse_norm_df
        else:
            logger.error(f'Status code backend server: {req.status_code}')
            logger.error(req.status_code)
            return None
    except Exception as e:
        logger.error(e)


# Перед подачей на эндпоинт нужно все превратить в json
json_list_general_norm_df = df_init.to_dict(orient='records')


# Пример вызова Time2Vec
logger.info("Normalizing the data.")
df_general_norm_df, min_val, max_val = normalization_request(
    col_time='datetime',
    col_target=measurement,
    json_list_df=json_list_general_norm_df
)

print(df_general_norm_df.head())
print(f'Колонки после нормализации - {df_general_norm_df.columns}')


# TODO Здесь может быть твой код прогноза


# Перед подачей на эндпоинт нужно все превратить в json
json_list_df_predict_norm = df_general_norm_df.to_dict(orient='records')


# Пример возвращения из вектора в дату
df_predict = reverse_normalization_request(
    col_time='datetime',
    col_target=measurement,
    json_list_norm_df=json_list_df_predict_norm,
    min_val=min_val,
    max_val=max_val
)
print(df_predict.head())
print(f'Колонки после обратной нормализации - {df_predict.columns}')
