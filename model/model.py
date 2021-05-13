import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, Response
import io
import pickle5
from pydantic import BaseModel

df_merge = pd.read_csv('res/df_2018.csv')
df_merge = df_merge.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
df_merge.sort_values(by=['Province', 'Year', 'Month', 'Day', 'UTC Hour'], inplace=True)

def get_interpolate(df):

  method = 'linear'
  limit_direction = 'both'
  axis = 1

  start = str(df.iloc[0].Time)
  end = str(df.iloc[-1].Time)

  # index_range = pd.date_range(start = start, end = end,freq='1H')
  # df = df.reindex(index_range)

  for column in list(df.columns):
    if df[column].dtype == object:
      df[column] = df[column].fillna(df[column].mode())

    elif df[column].dtype == 'int64':
      df[column] = df[column].astype('float64')
    
    if df[column].dtype == 'float64': 
      df[column] = df[column].interpolate(method = method,
                                        limit_direction = limit_direction,
                                        axis = 0)
    
  return df

# Group by province and interpolate by date-time

province_group = df_merge.groupby('Province')
provinces = list(df_merge.Province.value_counts().index)

sorted_df = []
for province in provinces:
  interpolated_df = get_interpolate(province_group.get_group(province))
  sorted_df.append(interpolated_df)

sorted_df = pd.concat(sorted_df)
sorted_df.drop_duplicates(subset=['Province', 'Year', 'Month', 'Day', 'UTC Hour'], keep='first', inplace=True)

df_merge_2018 = sorted_df[sorted_df['Year'] == 2018]

# Clear memory
del interpolated_df
del sorted_df
del df_merge

# Drop date columns
def drop_columns(x):
  drop_columns = ['date_time', 'Time', 'satellite', 'frp', 'daynight', 'type', 'scan', 'confidence']
  x.drop(drop_columns, axis=1, inplace=True)


drop_columns(df_merge_2018)

# Imputation of missing values for categories in pandas
def impute_missing_value(x):
  x.fillna(x.mode().iloc[0], inplace=True)


impute_missing_value(df_merge_2018)

# Separate nominal and ordinal columns
df_merge_ordinal = df_merge_2018.select_dtypes(exclude='object')
ordinal_columns = list(df_merge_ordinal.columns)

df_merge_nominal = df_merge_2018.select_dtypes(include='object')
nominal_columns = list(df_merge_nominal.drop(['index'], axis=1).columns)

# Convert nominal columns
import tensorflow as tf

converter = {}

def get_converter(labels):
  label_to_idx = {}
  idx_to_label = {}

  for x, y in enumerate(np.unique(labels)):
    label_to_idx[y] = x
    idx_to_label[x] = y
  return label_to_idx, idx_to_label

def get_nominal_dataset(dfx):
  nominal_dataset = []
  max_seq_length = 60

  for column in nominal_columns:
    dfx[column] = dfx[column].astype(str)
    tag_to_idx, idx_to_tag = get_converter(dfx[column])
    converter[column] = (tag_to_idx, idx_to_tag)
    dfx[column] = dfx[column].map(tag_to_idx)
    max_seq_length = max(max_seq_length, len(tag_to_idx))

  for column in nominal_columns:
    one_hot_vector = tf.keras.utils.to_categorical(dfx[column], max_seq_length)
    nominal_dataset.append(one_hot_vector)

  nominal_dataset = np.array(nominal_dataset)
  nominal_dataset = np.swapaxes(nominal_dataset, 0, 1)
  return nominal_dataset

nominal_dataset_2018 = get_nominal_dataset(df_merge_2018)

# Convert ordinal columns
def get_ordinal_dataset(dfx):
  ordinal_dataset = []

  for column in ordinal_columns:
    if dfx[column].dtype == 'int64':
      dfx[column] = dfx[column].astype('float64')
    vector = np.expand_dims(dfx[column], axis=1)
    ordinal_dataset.append(dfx[column])

  ordinal_dataset = np.swapaxes(ordinal_dataset, 0, 1)
  return ordinal_dataset

ordinal_dataset_2018 = get_ordinal_dataset(df_merge_2018)

df_2018 = df_merge_2018.reset_index(drop=True)

x_1_test = nominal_dataset_2018[:-72]
x_2_test = ordinal_dataset_2018[:-72]
y_test = df_merge_2018['PM2.5'][72:]

y_test.reset_index(inplace=True, drop=True)

model = tf.keras.models.load_model('res/model_1.h5')
converter = pickle5.load(open("res/converter.p", "rb"))
(province_2_idx, idx_2_province) = converter['Province']

def get_key(dfx, time, province):
  filter = (dfx['index'] == time) & (dfx['Province'] == province)
  # return dfx[filter].index
  return dfx[filter].index[0]


def predict(input_time, input_province):
    province_idx = province_2_idx[input_province]
    
    key = get_key(df_2018, input_time, province_idx)

    x_1 = np.array([x_1_test[key]])
    x_2 = np.array([x_2_test[key]])
    y = np.array([y_test[key]])

    predicted = model.predict([x_1, x_2]).flatten()
    actual = y

    res = { "predict": predicted.tolist(), "actual": actual.tolist()}
    return json.dumps(res)

def get_provinces():
    return province_2_idx.keys()