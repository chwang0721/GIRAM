import pickle as pkl
from os.path import join

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected


def get_seq_data(df):
    users = []
    poi_inputs = []
    timestamp_inputs = []
    location_inputs = []

    hour_inputs = []
    weekday_inputs = []
    norm_time_inputs = []
    main_cat_inputs = []
    cat_inputs = []
    region_inputs = []

    poi_labels = []
    max_len = 100 + 1
    traj_ids = []

    for traj_id in set(df['TrajId'].tolist()):
        traj_df = df[df['TrajId'] == traj_id]

        user_id = traj_df['UserId'].iloc[0]
        poi_seq = traj_df['PoiId'].to_list()
        timestamp_seq = traj_df['Timestamp'].tolist()

        location_seq = list(zip(traj_df['Latitude'], traj_df['Longitude']))
        hour_seq = traj_df['Hour'].tolist()
        weekday_seq = traj_df['Weekday'].tolist()
        norm_time_seq = traj_df['NormTime'].tolist()
        main_cat_seq = traj_df['PoiMainCatId'].tolist()
        cat_seq = traj_df['PoiCatId'].tolist()
        region_seq = traj_df['GridId'].tolist()

        if len(poi_seq) > max_len:
            poi_seq = poi_seq[-max_len:]
            timestamp_seq = timestamp_seq[-max_len:]
            location_seq = location_seq[-max_len:]

            hour_seq = hour_seq[-max_len:]
            weekday_seq = weekday_seq[-max_len:]
            norm_time_seq = norm_time_seq[-max_len:]
            main_cat_seq = main_cat_seq[-max_len:]
            cat_seq = cat_seq[-max_len:]
            region_seq = region_seq[-max_len:]

        for i in range(2, len(poi_seq)):
            users.append(user_id)
            poi_inputs.append(poi_seq[:i])
            timestamp_inputs.append(timestamp_seq[:i])
            location_inputs.append(location_seq[:i])
            hour_inputs.append(hour_seq[:i])
            weekday_inputs.append(weekday_seq[:i])
            norm_time_inputs.append(norm_time_seq[:i])
            main_cat_inputs.append(main_cat_seq[:i])
            cat_inputs.append(cat_seq[:i])
            region_inputs.append(region_seq[:i])
            poi_labels.append(poi_seq[i])
            traj_ids.append(traj_id)

    return (users, poi_inputs, timestamp_inputs, location_inputs, hour_inputs, weekday_inputs,
            norm_time_inputs, main_cat_inputs, cat_inputs, region_inputs, poi_labels, traj_ids)


def get_seq_data_test(df):
    users = []
    poi_inputs = []
    timestamp_inputs = []
    location_inputs = []

    hour_inputs = []
    weekday_inputs = []
    norm_time_inputs = []
    main_cat_inputs = []
    cat_inputs = []
    region_inputs = []

    poi_labels = []
    max_len = 100 + 1
    traj_ids = []

    for traj_id in set(df['TrajId'].tolist()):
        traj_df = df[df['TrajId'] == traj_id]

        user_id = traj_df['UserId'].iloc[0]
        poi_seq = traj_df['PoiId'].to_list()
        timestamp_seq = traj_df['Timestamp'].tolist()

        location_seq = list(zip(traj_df['Latitude'], traj_df['Longitude']))
        hour_seq = traj_df['Hour'].tolist()
        weekday_seq = traj_df['Weekday'].tolist()
        norm_time_seq = traj_df['NormTime'].tolist()
        main_cat_seq = traj_df['PoiMainCatId'].tolist()
        cat_seq = traj_df['PoiCatId'].tolist()
        region_seq = traj_df['GridId'].tolist()

        if len(poi_seq) > max_len:
            poi_seq = poi_seq[-max_len:]
            timestamp_seq = timestamp_seq[-max_len:]
            location_seq = location_seq[-max_len:]

            hour_seq = hour_seq[-max_len:]
            weekday_seq = weekday_seq[-max_len:]
            norm_time_seq = norm_time_seq[-max_len:]
            main_cat_seq = main_cat_seq[-max_len:]
            cat_seq = cat_seq[-max_len:]
            region_seq = region_seq[-max_len:]

        users.append(user_id)
        poi_inputs.append(poi_seq[:-1])
        timestamp_inputs.append(timestamp_seq[:-1])
        location_inputs.append(location_seq[:-1])
        hour_inputs.append(hour_seq[:-1])
        weekday_inputs.append(weekday_seq[:-1])
        norm_time_inputs.append(norm_time_seq[:-1])
        main_cat_inputs.append(main_cat_seq[:-1])
        cat_inputs.append(cat_seq[:-1])
        region_inputs.append(region_seq[:-1])
        poi_labels.append(poi_seq[-1])
        traj_ids.append(traj_id)

    return (users, poi_inputs, timestamp_inputs, location_inputs, hour_inputs, weekday_inputs,
            norm_time_inputs, main_cat_inputs, cat_inputs, region_inputs, poi_labels, traj_ids)


def getSeqGraph(seq, time_list, dist_mat):
    i, x, senders, nodes = 0, [], [], {}
    for node in seq:
        if node not in nodes:
            nodes[node] = i
            x.append([node])
            i += 1
        senders.append(nodes[node])
    x = torch.LongTensor(x)
    edge_index = torch.LongTensor([senders[: -1], senders[1:]])

    def get_min(interv):
        interv_min = interv.clone()
        interv_min[interv_min == 0] = 2 ** 31
        return interv_min.min()

    time_interv = (time_list[1:] - time_list[:-1]).long()
    dist_interv = dist_mat[seq[:-1], seq[1:]].long()
    mean_interv = dist_interv.float().mean()
    if time_interv.size(0) > 0:
        time_interv = torch.clamp((time_interv / get_min(time_interv)).long(), 0, 256 - 1)
        dist_interv = torch.clamp((dist_interv / get_min(dist_interv)).long(), 0, 256 - 1)
    return Data(x=x, edge_index=edge_index, num_nodes=len(nodes), mean_interv=mean_interv, edge_time=time_interv,
                edge_dist=dist_interv)


class GraphData(Dataset):
    def __init__(self, n_user, n_poi, seq_data, dist_mat, tr_dict):
        self.n_user, self.n_poi = n_user, n_poi
        self.seq_data = seq_data
        self.dist_mat = dist_mat
        self.tr_dict = tr_dict

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, index):
        (users, poi_inputs, timestamp_inputs, location_inputs, hour_inputs, weekday_inputs,
         norm_time_inputs, main_cat_inputs, cat_inputs, region_inputs, poi_labels, traj_ids) = self.seq_data[index]
        timestamp_inputs = torch.LongTensor(timestamp_inputs)
        seq_graph = getSeqGraph(poi_inputs, timestamp_inputs, self.dist_mat)
        poi_inputs = torch.LongTensor(poi_inputs)

        exclude_mask = torch.zeros((self.n_poi,)).bool()
        if users in self.tr_dict:
            exclude_set = torch.LongTensor(list(set(self.tr_dict[users])))
            exclude_mask[exclude_set] = 1
        exclude_mask[poi_labels] = 0
        return seq_graph, poi_inputs, timestamp_inputs, location_inputs, hour_inputs, weekday_inputs, \
                norm_time_inputs, main_cat_inputs, cat_inputs, region_inputs, poi_labels, exclude_mask, users, traj_ids


def get_geo_graph(poi_num, dataset):
    dist_pth = f'data/{dataset}'
    with open(join(dist_pth, 'dist_graph.pkl'), 'rb') as f:
        geo_edges = torch.LongTensor(pkl.load(f))
    edge_weights = torch.Tensor(np.load(join(dist_pth, 'dist_on_graph.npy')))
    edge_weights /= edge_weights.max()
    geo_edges, edge_weights = to_undirected(geo_edges, edge_weights, num_nodes=poi_num)
    assert geo_edges.size(1) == edge_weights.size(0)
    geo_graph = Data(edge_index=geo_edges, edge_attr=edge_weights)
    return geo_graph


def collate_fn(batch):
    (seq_graph, poi_inputs, timestamp_inputs, location_inputs, hour_inputs, weekday_inputs, norm_time_inputs,
     main_cat_inputs, cat_inputs, region_inputs, poi_labels, exclude_mask, users, traj_ids) = zip(*batch)
    seq_graph = Batch.from_data_list(seq_graph)
    poi_labels = torch.LongTensor(poi_labels)

    lengths = [len(poi) for poi in poi_inputs]
    max_length = max(lengths)
    hour_inputs = torch.LongTensor([seq + [0] * (max_length - len(seq)) for seq in hour_inputs])
    weekday_inputs = torch.LongTensor([seq + [0] * (max_length - len(seq)) for seq in weekday_inputs])
    norm_time_inputs = torch.Tensor([seq + [0] * (max_length - len(seq)) for seq in norm_time_inputs])
    location_inputs = torch.Tensor([seq + [(0, 0)] * (max_length - len(seq)) for seq in location_inputs])
    main_cat_inputs = torch.LongTensor([seq + [0] * (max_length - len(seq)) for seq in main_cat_inputs])
    cat_inputs = torch.LongTensor([seq + [0] * (max_length - len(seq)) for seq in cat_inputs])
    region_inputs = torch.LongTensor([seq + [0] * (max_length - len(seq)) for seq in region_inputs])
    timestamp_inputs = torch.LongTensor([seq.tolist() + [0] * (max_length - len(seq)) for seq in timestamp_inputs])

    lengths = torch.LongTensor(lengths)

    return (seq_graph, poi_inputs, poi_labels, timestamp_inputs, hour_inputs, weekday_inputs, norm_time_inputs,
            location_inputs, main_cat_inputs, cat_inputs, region_inputs, lengths, users, exclude_mask)


def load_data(dataset, idx, user_num, poi_num):
    df = pd.read_csv(f'data/{dataset}/{idx}.csv')
    if idx == 0:
        idx = 1

    csv_files = [f"data/{dataset}/{i}.csv" for i in range(idx - 1, -1, -1)]  # Create a list of file paths

    df_train = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)  # Efficient concatenation
    trajs = list(zip(*get_seq_data(df_train)))
    train_dict = {}
    for traj in trajs:
        user_id = traj[0]
        poi_seq = traj[1]
        if user_id not in train_dict:
            train_dict[user_id] = []
        train_dict[user_id].extend(poi_seq)

    dist_mat = torch.from_numpy(np.load(f'data/{dataset}/dist_mat.npy'))
    ds = GraphData(user_num, poi_num, list(zip(*get_seq_data(df))), dist_mat, train_dict)
    return ds


def load_testdata(dataset, idx, user_num, poi_num):
    df = pd.read_csv(f'data/{dataset}/{idx}.csv')
    csv_files = [f"data/{dataset}/{i}.csv" for i in range(idx - 1, -1, -1)]  # Create a list of file paths

    df_train = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)  # Efficient concatenation
    trajs = list(zip(*get_seq_data(df_train)))
    train_dict = {}
    for traj in trajs:
        user_id = traj[0]
        poi_seq = traj[1]
        if user_id not in train_dict:
            train_dict[user_id] = []
        train_dict[user_id].extend(poi_seq)

    dist_mat = torch.from_numpy(np.load(f'data/{dataset}/dist_mat.npy'))
    ds = GraphData(user_num, poi_num, list(zip(*get_seq_data_test(df))), dist_mat, train_dict)
    return ds


def load_data_ader(dataset, idx, user_num, poi_num):
    df = pd.read_csv(f'data/{dataset}/{idx}.csv')

    csv_files = [f"data/{dataset}/{i}.csv" for i in range(idx, -1, -1)]  # Create a list of file paths

    df_train = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)  # Efficient concatenation
    trajs = list(zip(*get_seq_data(df_train)))
    train_dict = {}
    for traj in trajs:
        user_id = traj[0]
        poi_seq = traj[1]
        if user_id not in train_dict:
            train_dict[user_id] = []
        train_dict[user_id].extend(poi_seq)

    dist_mat = torch.from_numpy(np.load(f'data/{dataset}/dist_mat.npy'))
    ds = GraphData(user_num, poi_num, list(zip(*get_seq_data_test(df))), dist_mat, train_dict)
    return ds


def pretrain_dataloader(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader

def dataloader(train_dataset, next_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    split_size = len(next_dataset) // 2
    valid_dataset, test_dataset = random_split(next_dataset, [split_size, len(next_dataset) - split_size])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader


def get_all_seq_data(df, ids):
    users = []
    poi_inputs = []
    timestamp_inputs = []
    location_inputs = []

    hour_inputs = []
    weekday_inputs = []
    norm_time_inputs = []
    main_cat_inputs = []
    cat_inputs = []
    region_inputs = []

    poi_labels = []
    max_len = 100 + 1
    traj_ids = []

    for traj_id in ids:
        traj_df = df[df['TrajId'] == traj_id]

        user_id = traj_df['UserId'].iloc[0]
        poi_seq = traj_df['PoiId'].to_list()
        timestamp_seq = traj_df['Timestamp'].tolist()

        location_seq = list(zip(traj_df['Latitude'], traj_df['Longitude']))
        hour_seq = traj_df['Hour'].tolist()
        weekday_seq = traj_df['Weekday'].tolist()
        norm_time_seq = traj_df['NormTime'].tolist()
        main_cat_seq = traj_df['PoiMainCatId'].tolist()
        cat_seq = traj_df['PoiCatId'].tolist()
        region_seq = traj_df['GridId'].tolist()

        if len(poi_seq) > max_len:
            poi_seq = poi_seq[-max_len:]
            timestamp_seq = timestamp_seq[-max_len:]
            location_seq = location_seq[-max_len:]

            hour_seq = hour_seq[-max_len:]
            weekday_seq = weekday_seq[-max_len:]
            norm_time_seq = norm_time_seq[-max_len:]
            main_cat_seq = main_cat_seq[-max_len:]
            cat_seq = cat_seq[-max_len:]
            region_seq = region_seq[-max_len:]

        for i in range(2, len(poi_seq)):
            users.append(user_id)
            poi_inputs.append(poi_seq[:i])
            timestamp_inputs.append(timestamp_seq[:i])
            location_inputs.append(location_seq[:i])
            hour_inputs.append(hour_seq[:i])
            weekday_inputs.append(weekday_seq[:i])
            norm_time_inputs.append(norm_time_seq[:i])
            main_cat_inputs.append(main_cat_seq[:i])
            cat_inputs.append(cat_seq[:i])
            region_inputs.append(region_seq[:i])
            poi_labels.append(poi_seq[i])
            traj_ids.append(traj_id)

    return (users, poi_inputs, timestamp_inputs, location_inputs, hour_inputs, weekday_inputs,
            norm_time_inputs, main_cat_inputs, cat_inputs, region_inputs, poi_labels, traj_ids)


def get_all_trajs(trajs, dataset, idx, user_num, poi_num):
    df = pd.DataFrame()
    for i in range(0, idx + 1):
        df_i = pd.read_csv(f'data/{dataset}/{i}.csv')
        df = pd.concat([df, df_i])
    dist_mat = torch.from_numpy(np.load(f'data/{dataset}/dist_mat.npy'))
    o_traj_ids = []
    for traj in trajs:
        o_traj_ids.append(traj[-1])

    trajs = list(zip(*get_seq_data(df)))
    train_dict = {}
    for traj in trajs:
        user_id = traj[0]
        poi_seq = traj[1]
        if user_id not in train_dict:
            train_dict[user_id] = []
        train_dict[user_id].extend(poi_seq)

    ds = GraphData(user_num, poi_num, list(zip(*get_all_seq_data(df, o_traj_ids))), dist_mat, train_dict)
    return ds
