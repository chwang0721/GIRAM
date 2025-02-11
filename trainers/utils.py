import numpy as np
import pandas as pd
import torch


def top_k_acc(y_true, y_pred, k):
    top_k_preds = torch.topk(y_pred, k, dim=1).indices  # [batch_size, k]
    correct = top_k_preds.eq(y_true.unsqueeze(1))  # [batch_size, k]
    accuracy = correct.any(dim=1).float().sum().item()
    return accuracy


def mean_reciprocal_rank(y_true, y_pred):
    sorted_indices = torch.argsort(y_pred, dim=1, descending=True)  # [batch_size, num_classes]
    true_indices = (sorted_indices == y_true.unsqueeze(1)).nonzero(as_tuple=False)  # [batch_size, 2]
    ranks = true_indices[:, 1] + 1
    reciprocal_ranks = 1.0 / ranks.float()  # [batch_size]
    mrr = reciprocal_ranks.sum().item()
    return mrr


def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='PoiCatId',
                             feature3='Latitude', feature4='Longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X


def calculate_laplacian_matrix(adj_mat):
    n_vertex = adj_mat.shape[0]
    deg_mat = torch.diag(torch.sum(adj_mat, dim=1))
    id_mat = torch.eye(n_vertex, device=adj_mat.device)
    wid_deg_mat = deg_mat + id_mat
    wid_adj_mat = adj_mat + id_mat

    wid_deg_mat_inv = torch.linalg.inv(wid_deg_mat)
    hat_rw_normd_lap_mat = torch.matmul(wid_deg_mat_inv, wid_adj_mat)

    return hat_rw_normd_lap_mat


def make_mask(input_seqs, lengths):
    mask = torch.zeros_like(input_seqs)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return mask
