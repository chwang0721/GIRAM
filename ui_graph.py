import os
import pickle as pkl
from math import cos, asin, sqrt, pi
from os.path import join

import numpy as np
import pandas as pd
import torch

from configs.config import args


def distance_mat_form(lat_vec, lon_vec):
    # Shape of lat_vec & lon_vec: [poi_num, 1]
    r = 6371
    p = pi / 180
    lat_mat = np.repeat(lat_vec, lat_vec.shape[0], axis=-1)
    lon_mat = np.repeat(lon_vec, lon_vec.shape[0], axis=-1)
    a_mat = 0.5 - np.cos((lat_mat.T - lat_mat) * p) / 2 \
            + np.matmul(np.cos(lat_vec * p), np.cos(lat_vec * p).T) * (1 - np.cos((lon_mat.T - lon_mat) * p)) / 2
    return 2 * r * np.arcsin(np.sqrt(a_mat))


def gen_nei_graph(df, user_nums):
    nei_dict = {idx: [] for idx in range(user_nums)}
    edges = [[], []]

    for _uid, _item in df.groupby('UserId'):
        poi_list = _item['PoiId'].tolist()

        nei_dict[_uid] = poi_list
        edges[0] += [_uid for _ in poi_list]
        edges[1] += poi_list

    return nei_dict, torch.LongTensor(edges)


def gen_loc_graph(poi_loc, poi_num, thre):

    lat_vec = np.array([poi_loc[_poi][0] for _poi in range(poi_num)], dtype=np.float64).reshape(-1, 1)
    lon_vec = np.array([poi_loc[_poi][1] for _poi in range(poi_num)], dtype=np.float64).reshape(-1, 1)
    _dist_mat = distance_mat_form(lat_vec, lon_vec)

    adj_mat = np.triu(_dist_mat <= thre, k=1)
    num_edges = adj_mat.sum()
    print(f'Edges on dist_graph: {num_edges}, avg degree: {num_edges / poi_num}')

    idx_mat = np.arange(poi_num).reshape(-1, 1).repeat(poi_num, -1)
    row_idx = idx_mat[adj_mat]
    col_idx = idx_mat.T[adj_mat]
    edges = np.stack((row_idx, col_idx))

    nei_dict = {poi: [] for poi in range(poi_num)}
    for e_idx in range(edges.shape[1]):
        src, dst = edges[:, e_idx]
        nei_dict[src].append(dst)
        nei_dict[dst].append(src)
    return _dist_mat, edges, nei_dict


if __name__ == '__main__':
    dataset = args.dataset
    dist_pth = f'./data/{dataset}'
    # Build POI checkin trajectory graph
    df = pd.read_csv(os.path.join(dist_pth, 'all.csv'))
    user_num = len(df['UserId'].unique())
    poi_num = len(df['PoiId'].unique())
    df_0 = pd.read_csv(os.path.join(dist_pth, '0.csv'))

    ui_nei_dict, ui_edges = gen_nei_graph(df, user_num)
    with open(join(dist_pth, 'ui_graph.pkl'), 'wb') as f:
        pkl.dump(ui_nei_dict, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(ui_edges, f, pkl.HIGHEST_PROTOCOL)

    dist_threshold = 1.
    print(f'UI graph dumped, generating location graph with delta d: {dist_threshold}km...')

    # get loc_dict
    loc_dict = np.load(join(dist_pth, 'loc_dict.npy'), allow_pickle=True).item()

    dist_mat, dist_edges, dist_dict = gen_loc_graph(loc_dict, poi_num, dist_threshold)
    with open(join(dist_pth, 'dist_graph.pkl'), 'wb') as f:
        pkl.dump(dist_edges, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(dist_dict, f, pkl.HIGHEST_PROTOCOL)
    np.save(join(dist_pth, 'dist_mat.npy'), dist_mat)

    dist_on_graph = dist_mat[dist_edges[0], dist_edges[1]]
    np.save(join(dist_pth, 'dist_on_graph.npy'), dist_on_graph)
    print('Distance graph dumped, process done.')
