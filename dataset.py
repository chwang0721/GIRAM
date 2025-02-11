import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split


class TrajectoryDataset:
    def __init__(self, df, max_len=101):
        self.users = []
        self.poi_inputs = []
        self.timestamp_inputs = []
        self.location_inputs = []

        self.hour_inputs = []
        self.weekday_inputs = []
        self.norm_time_inputs = []
        self.main_cat_inputs = []
        self.cat_inputs = []
        self.region_inputs = []

        self.poi_labels = []
        self.norm_time_labels = []
        self.cat_labels = []

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

            self.users.append(user_id)
            self.poi_inputs.append(poi_seq[:-1])
            self.timestamp_inputs.append(timestamp_seq[:-1])
            self.location_inputs.append(location_seq[:-1])

            self.hour_inputs.append(hour_seq[:-1])
            self.weekday_inputs.append(weekday_seq[:-1])
            self.norm_time_inputs.append(norm_time_seq[:-1])
            self.main_cat_inputs.append(main_cat_seq[:-1])
            self.cat_inputs.append(cat_seq[:-1])
            self.region_inputs.append(region_seq[:-1])

            self.poi_labels.append(poi_seq[1:])
            self.norm_time_labels.append(norm_time_seq[1:])
            self.cat_labels.append(cat_seq[1:])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return (self.users[index], self.poi_inputs[index], self.timestamp_inputs[index], self.location_inputs[index],
                self.norm_time_inputs[index], self.hour_inputs[index], self.weekday_inputs[index],
                self.main_cat_inputs[index], self.cat_inputs[index], self.region_inputs[index], self.poi_labels[index],
                self.norm_time_labels[index], self.cat_labels[index])


def load_data(dataset, idx):
    df = pd.read_csv(f'data/{dataset}/{idx}.csv')
    data = TrajectoryDataset(df)
    del df
    return data


def collate_fn(batch):
    max_len = 100
    (users, poi_inputs, timestamp_inputs, location_inputs, norm_time_inputs, hour_inputs, weekday_inputs,
     main_cat_inputs, cat_inputs, region_inputs, poi_labels, norm_time_labels, cat_labels) = zip(*batch)
    lengths = [len(poi_inputs) for poi_inputs in poi_inputs]

    users = torch.LongTensor(users)
    poi_inputs = torch.LongTensor([seq + [0] * (max_len - len(seq)) for seq in poi_inputs])
    timestamp_inputs = torch.LongTensor([seq + [0] * (max_len - len(seq)) for seq in timestamp_inputs])
    location_inputs = torch.Tensor([seq + [(0, 0)] * (max_len - len(seq)) for seq in location_inputs])

    hour_inputs = torch.LongTensor([seq + [0] * (max_len - len(seq)) for seq in hour_inputs])
    weekday_inputs = torch.LongTensor([seq + [0] * (max_len - len(seq)) for seq in weekday_inputs])
    norm_time_inputs = torch.Tensor([seq + [0] * (max_len - len(seq)) for seq in norm_time_inputs])
    main_cat_inputs = torch.LongTensor([seq + [0] * (max_len - len(seq)) for seq in main_cat_inputs])
    cat_inputs = torch.LongTensor([seq + [0] * (max_len - len(seq)) for seq in cat_inputs])
    region_inputs = torch.LongTensor([seq + [0] * (max_len - len(seq)) for seq in region_inputs])

    poi_labels = torch.LongTensor([seq + [0] * (max_len - len(seq)) for seq in poi_labels])
    norm_time_labels = torch.Tensor([seq + [0] * (max_len - len(seq)) for seq in norm_time_labels])
    cat_labels = torch.LongTensor([seq + [0] * (max_len - len(seq)) for seq in cat_labels])

    lengths = torch.LongTensor(lengths)

    return (users, poi_inputs, timestamp_inputs, location_inputs, hour_inputs, weekday_inputs, norm_time_inputs,
            main_cat_inputs, cat_inputs, region_inputs, poi_labels, norm_time_labels, cat_labels, lengths)


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
