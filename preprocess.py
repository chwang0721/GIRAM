import ast
import json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from configs.config import args


# Extract POI category name for Gowalla dataset
def extract_name(json_str):
    try:
        parsed = ast.literal_eval(json_str)
        return parsed[0]['name'] if parsed and 'name' in parsed[0] else None
    except (ValueError, SyntaxError, KeyError):
        return None


# Read dataset
def read_dataset(file_path, dataset):
    if dataset == 'NYC' or dataset == 'TKY':
        df = pd.read_csv(file_path, sep='\t', encoding='latin-1', header=None)
        df.columns = ['UserId', 'PoiId', 'PoiCatId', 'PoiCatName', 'Latitude', 'Longitude', 'TimeOffset', 'UTCTime']
        df['UTCTime'] = df['UTCTime'].apply(lambda x: datetime.strptime(x, '%a %b %d %H:%M:%S +0000 %Y'))
        df['Time'] = df['UTCTime'] + df['TimeOffset'].apply(lambda x: timedelta(hours=x / 60))
        grid_size = 0.03

    elif dataset == 'Gowalla':
        df = pd.read_csv(file_path, sep=',')
        df['Time'] = df['UTCTime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
        df['PoiCatName'] = df['PoiCategoryId'].apply(extract_name)
        grid_size = 0.3

    df['Timestamp'] = df['Time'].apply(lambda x: x.timestamp())
    df['Hour'] = df['Time'].apply(lambda x: x.hour)
    df['Weekday'] = df['Time'].apply(lambda x: x.weekday())
    df['NormTime'] = df['Time'].apply(lambda x: (x.hour * 3600 + x.minute * 60 + x.second) / 86400)
    df['GridX'] = (df['Latitude'] / grid_size).astype(int)
    df['GridY'] = (df['Longitude'] / grid_size).astype(int)
    df['Grid'] = df['GridX'].astype(str) + '_' + df['GridY'].astype(str)

    return df


# Filter out users and POIs with low frequency
def do_filter(df, poi_min_freq=10, user_min_freq=10):
    df = df.copy()
    df['PoiFreq'] = df.groupby('PoiId')['UserId'].transform('count')
    df = df[df['PoiFreq'] >= poi_min_freq]
    df['UserFreq'] = df.groupby('UserId')['PoiId'].transform('count')
    df = df[df['UserFreq'] >= user_min_freq]
    df = df.drop(columns=['PoiFreq', 'UserFreq'])
    return df


# Split the dataset into 1 base block and 5 incremental blocks
def split_incremental(df):
    df = df.sort_values(by='Time', ascending=True).reset_index(drop=True)
    base_index = int(df.shape[0] * 0.5)
    df_all = dict()
    df_all[0] = df[:base_index]
    num_each_block = (df.shape[0] - base_index) // block_num
    for i in range(1, block_num + 1):
        df_incremental = df[base_index + (i - 1) * num_each_block: base_index + i * num_each_block]
        df_all[i] = df_incremental
    return df_all


# Generate trajectory id, each trajectory has a maximum time interval of 7 days
def generate_traj_id(df, traj_time_interval=3600 * 24 * 7):
    df = df.sort_values(by=['UserId', 'Time'], ascending=True).reset_index(drop=True)
    traj_id = 0
    start_time = None
    last_user = None
    traj_ids = []

    for idx, row in df.iterrows():
        current_user = row['UserId']
        current_time = row['Time']
        if last_user != current_user:
            traj_id += 1
            start_time = current_time
        elif (current_time - start_time).total_seconds() > traj_time_interval:
            traj_id += 1
            start_time = current_time
        traj_ids.append(traj_id)
        last_user = current_user

    df['TrajId'] = traj_ids

    # Filter out trajectories with only one check-in
    traj_counts = df['TrajId'].value_counts()
    valid_traj_ids = traj_counts[traj_counts >= 2].index
    df = df[df['TrajId'].isin(valid_traj_ids)]
    return df


# Encode user id, POI id, and POI category id
def encode_poi(df):
    df['UserId'] = df['UserId'].astype('category').cat.codes
    df['PoiId'] = df['PoiId'].astype('category').cat.codes
    df['PoiMainCatId'] = df['PoiMainCatName'].astype('category').cat.codes
    df['PoiCatId'] = df['PoiCatName'].astype('category').cat.codes
    df['GridId'] = df['Grid'].astype('category').cat.codes
    return df

def split_valid_test(df):
    # randomly split the dataset into validation and test sets
    valid_index = df.groupby('UserId').apply(lambda x: x.sample(frac=0.5, random_state=1)).index
    df_valid = df[df.index.isin(valid_index)]
    df_test = df[~df.index.isin(valid_index)]
    return df_valid, df_test


# Print dataset statistics
def dataset_statistics(df, traj_num):
    print('======================================')
    print('Dataset:', dataset)
    print('Number of users:', df['UserId'].nunique())
    print('Number of POIs:', df['PoiId'].nunique())
    print('Number of main POI categories:', df['PoiMainCatId'].nunique())
    print('Number of POI categories:', df['PoiCatId'].nunique())
    print('Number of grids:', df['GridId'].nunique())
    print('Number of check-ins:', len(df))
    print('Number of trajectories:', traj_num)
    print('======================================')


if __name__ == '__main__':
    dataset = args.dataset
    if dataset == 'NYC':
        file_path = 'data/NYC/dataset_TSMC2014_NYC.txt'
    elif dataset == 'TKY':
        file_path = 'data/TKY/dataset_TSMC2014_TKY.txt'
    elif dataset == 'Gowalla':
        file_path = 'data/Gowalla/dataset_gowalla_ca_ne.csv'

    df = read_dataset(file_path, dataset)
    df = do_filter(df)
    block_num = 5

    # Extract main POI category, cat.json file is obtained from ChatGPT
    cat_names = json.load(open(f'data/{dataset}/cats.json', 'r'))
    df['PoiMainCatName'] = df['PoiCatName'].apply(lambda x: next((k for k, v in cat_names.items() if x in v), None))

    df = split_incremental(df)

    traj_num = 0
    df_all = pd.DataFrame()
    for i in range(block_num + 1):
        df_i = generate_traj_id(df[i])
        df_i['block'] = i
        df_all = pd.concat([df_all, df_i])
        traj_num += df_i['TrajId'].nunique()

    df_all = encode_poi(df_all)
    dataset_statistics(df_all, traj_num)

    df_all.to_csv(f'data/{dataset}/all.csv', index=False)
    for i in range(block_num + 1):
        df_i = df_all[df_all['block'] == i]
        df_i.to_csv(f'data/{dataset}/{i}.csv', index=False)

    loc_dict = {poi: None for poi in range(len(df_all['PoiId'].unique()))}
    for poi, item in df_all.groupby('PoiId'):
        lat, lon = item['Latitude'].iloc[0], item['Longitude'].iloc[0]
        loc_dict[poi] = (lat, lon)
    np.save(f'data/{dataset}/loc_dict.npy', np.array(loc_dict, dtype=object))
