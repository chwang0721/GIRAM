import os

import networkx as nx
import numpy as np
import pandas as pd

from configs.config import args


def build_global_POI_checkin_graph(df_0, user_num, df):
    G = nx.DiGraph()
    users = range(user_num)
    for UserId in users:
        user_df = df_0[df_0['UserId'] == UserId]
        if user_df.shape[0] == 0:
            continue
        # Add nodes
        for i, row in user_df.iterrows():
            node = row['PoiId']
            if node not in G.nodes():
                G.add_node(row['PoiId'],
                           checkin_cnt=1,
                           PoiCatId=row['PoiCatId'],
                           Latitude=row['Latitude'],
                           Longitude=row['Longitude'])
            else:
                G.nodes[node]['checkin_cnt'] += 1

        # Add edges (Check-in seq)
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['PoiId']
            traj_id = row['TrajId']
            # No edge for the beginning of the seq or different traj
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue

            # Add edges
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id, weight=1)
            previous_traj_id = traj_id
            previous_poi_id = poi_id

    all_nodes = df['PoiId'].unique()
    for node in all_nodes:
        if node not in G.nodes():
            G.add_node(node,
                       checkin_cnt=0,
                       PoiCatId=df[df['PoiId'] == node]['PoiCatId'].iloc[0],
                       Latitude=df[df['PoiId'] == node]['Latitude'].iloc[0],
                       Longitude=df[df['PoiId'] == node]['Longitude'].iloc[0])

    return G


def save_graph_to_csv(G, dst_dir):
    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    np.savetxt(os.path.join(dst_dir, 'graph_A.csv'), A.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]
    with open(os.path.join(dst_dir, 'graph_X.csv'), 'w') as f:
        print('PoiID,checkin_cnt,PoiCatId,Latitude,Longitude', file=f)
        for each in nodes_data:
            node_name = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            poi_catid = each[1]['PoiCatId']
            latitude = each[1]['Latitude']
            longitude = each[1]['Longitude']
            print(f'{node_name}, {checkin_cnt}, 'f'{poi_catid},'
                  f'{latitude}, {longitude}', file=f)


if __name__ == '__main__':
    dst_dir = f'data/{args.dataset}'

    # Build POI checkin trajectory graph
    df_0 = pd.read_csv(os.path.join(dst_dir, '0.csv'))
    df = pd.read_csv(os.path.join(dst_dir, 'all.csv'))
    user_num = df['UserId'].nunique()
    G = build_global_POI_checkin_graph(df_0, user_num, df)

    # Save graph to disk
    save_graph_to_csv(G, dst_dir=dst_dir)
    print(f'{args.dataset} graph saved.')
