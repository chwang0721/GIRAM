import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import softmax
from torchsde import sdeint


class VP_SDE(nn.Module):
    def __init__(self, hid_dim, device, beta_min=0.1, beta_max=20, dt=1e-1):
        super(VP_SDE, self).__init__()
        self.hid_dim = hid_dim
        self.beta_min, self.beta_max = beta_min, beta_max
        self.dt = dt

        self.score_fn = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim)
        )
        for w in self.score_fn:
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

        self.device = device

    def calc_score(self, x, condition):
        return self.score_fn(torch.cat((x, condition), dim=-1))

    def forward_sde(self, x, t):
        def f(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            return -0.5 * beta_t * y

        def g(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs = y.size(0)
            noise = torch.ones((bs, self.hid_dim, 1), device=y.device)
            return (beta_t ** 0.5) * noise

        ts = torch.Tensor([0, t]).to(self.device)
        output = sdeint(SDEWrapper(f, g), y0=x, ts=ts, dt=self.dt)[-1]
        # output = x
        return output

    def get_beta_t(self, _t):
        return self.beta_min + _t * (self.beta_max - self.beta_min)

    def reverse_sde(self, x, condition, t):
        def f(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            score = self.score_fn(torch.cat((x, condition), dim=-1))
            drift = -0.5 * beta_t * y
            drift = drift - beta_t * score
            return drift

        def g(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs = y.size(0)
            noise = torch.ones((bs, self.hid_dim, 1), device=y.device)
            return (beta_t ** 0.5) * noise

        ts = torch.Tensor([0, t]).to(self.device)
        output = sdeint(SDEWrapper(f, g), y0=x, ts=ts, dt=self.dt)[-1]
        # output = x
        return output

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        log_mean_coeff = torch.Tensor([log_mean_coeff]).to(x.device)
        mean = torch.exp(log_mean_coeff.unsqueeze(-1)) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std


class SDEWrapper(nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'scalar'

    def __init__(self, f, g):
        super(SDEWrapper).__init__()
        self.f, self.g = f, g

    def f(self, t, y):
        return self.f(t, y)

    def g(self, t, y):
        return self.g(t, y)


class GeoConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GeoConv, self).__init__(aggr='add')
        self._cached_edge = None
        self.lin = nn.Linear(in_channels, out_channels)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, geo_graph: Data):
        if self._cached_edge is None:
            self._cached_edge = gcn_norm(geo_graph.edge_index, add_self_loops=False)
        edge_index, norm_weight = self._cached_edge
        x = self.lin(x)

        return self.propagate(edge_index, x=x, norm=norm_weight, dist_vec=geo_graph.edge_attr)

    def message(self, x_j, norm, dist_vec):
        return norm.unsqueeze(-1) * x_j * dist_vec.unsqueeze(-1)


class SeqConv(MessagePassing):
    def __init__(self, hid_dim, flow="source_to_target"):
        super(SeqConv, self).__init__(aggr='add', flow=flow)
        self.hid_dim = hid_dim
        self.alpha_src = nn.Linear(hid_dim, 1, bias=False)
        self.alpha_dst = nn.Linear(hid_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.alpha_src.weight)
        nn.init.xavier_uniform_(self.alpha_dst.weight)
        self.act = nn.LeakyReLU()

    def forward(self, embs, seq_graph):
        node_embs, distance_embs, temporal_embs = embs
        sess_idx, edge_index, batch_idx = seq_graph.x.squeeze(), seq_graph.edge_index, seq_graph.batch
        edge_time, edge_dist = seq_graph.edge_time, seq_graph.edge_dist

        x = node_embs[sess_idx]
        edge_l = distance_embs[edge_dist]
        edge_t = temporal_embs[edge_time]

        all_edges = torch.cat((edge_index, edge_index[[1, 0]]), dim=-1)
        seq_embs = self.propagate(all_edges, x=x, edge_l=edge_l, edge_t=edge_t, edge_size=edge_index.size(1))
        return seq_embs

    def message(self, x_j, x_i, edge_index_j, edge_index_i, edge_l, edge_t, edge_size):
        element_sim = x_j * x_i
        src_logits = self.alpha_src(element_sim[: edge_size] + edge_l + edge_t).squeeze(-1)
        dst_logits = self.alpha_dst(element_sim[edge_size:] + edge_l + edge_t).squeeze(-1)

        tot_logits = torch.cat((src_logits, dst_logits))
        attn_weight = softmax(tot_logits, edge_index_i)
        aggr_embs = x_j * attn_weight.unsqueeze(-1)
        return aggr_embs


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


def sequence_mask(lengths, max_len=None) -> torch.Tensor:
    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1))).reshape(lengths_shape)


class DiffPOI(nn.Module):
    def __init__(self, n_user, n_poi, geo_graph, args, model_type='u'):
        super(DiffPOI, self).__init__()
        self.n_user, self.n_poi = n_user, n_poi
        self.hid_dim = args.hidden
        self.step_num = 1000

        self.poi_emb = nn.Parameter(torch.empty(n_poi, self.hid_dim))
        self.distance_emb = nn.Parameter(torch.empty(args.interval, self.hid_dim))
        self.temporal_emb = nn.Parameter(torch.empty(args.interval, self.hid_dim))
        nn.init.xavier_normal_(self.poi_emb)
        nn.init.xavier_normal_(self.distance_emb)
        nn.init.xavier_normal_(self.temporal_emb)

        self.geo_encoder = GeoEncoder(n_poi, self.hid_dim, geo_graph, args)
        self.seq_encoder = SeqEncoder(self.hid_dim)
        self.sde = VP_SDE(self.hid_dim, args.device, dt=args.stepsize)
        self.ce_criteria = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=1 - args.keepprob)

        self.seq_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn = nn.MultiheadAttention(self.hid_dim, num_heads=2, batch_first=True, dropout=0.2)
        self.seq_forward = PointWiseFeedForward(self.hid_dim, 0.2)

        self.geo_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.geo_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.geo_attn = nn.MultiheadAttention(self.hid_dim, num_heads=2, batch_first=True, dropout=0.2)
        self.geo_forward = PointWiseFeedForward(self.hid_dim, 0.2)

        self.device = args.device
        self.diffsize = args.diffsize
        self.model_type = model_type

    def geoProp(self, poi_embs, seqs, seq_encs):
        geo_embs = self.geo_encoder.encode(poi_embs)
        geo_embs = self.dropout(geo_embs)

        seq_lengths = torch.LongTensor([seq.size(0) for seq in seqs]).to(self.device)
        geo_seq_embs = [geo_embs[seq] for seq in seqs]

        # Target-attention
        geo_embs_pad = pad_sequence(geo_seq_embs, batch_first=True, padding_value=0)
        qry_embs = self.geo_layernorm(seq_encs.detach().unsqueeze(1))
        pad_mask = sequence_mask(seq_lengths)

        geo_embs_pad, _ = self.geo_attn(qry_embs, geo_embs_pad, geo_embs_pad, key_padding_mask=~pad_mask)
        geo_embs_pad = geo_embs_pad.squeeze(1)
        geo_embs_pad = self.geo_attn_layernorm(geo_embs_pad)

        geo_encs = self.geo_forward(geo_embs_pad)
        return geo_encs, geo_embs

    def seqProp(self, poi_embs, seq_graph):
        seq_embs = self.seq_encoder.encode((poi_embs, self.distance_emb, self.temporal_emb), seq_graph)
        seq_embs = self.dropout(seq_embs)
        seq_lengths = torch.bincount(seq_graph.batch)
        seq_embs = torch.split(seq_embs, seq_lengths.cpu().numpy().tolist())

        # Self-attention
        seq_embs_pad = pad_sequence(seq_embs, batch_first=True, padding_value=0)
        qry_embs = self.seq_layernorm(seq_embs_pad)
        pad_mask = sequence_mask(seq_lengths)

        seq_embs_pad, _ = self.seq_attn(qry_embs, seq_embs_pad, seq_embs_pad, key_padding_mask=~pad_mask)
        seq_embs_pad = seq_embs_pad + qry_embs
        seq_embs_pad = self.seq_attn_layernorm(seq_embs_pad)

        seq_embs_pad = self.seq_forward(seq_embs_pad)
        seq_embs_pad = [seq[:seq_len] for seq, seq_len in zip(seq_embs_pad, seq_lengths)]

        seq_encs = torch.stack([seq.mean(dim=0) for seq in seq_embs_pad], dim=0)
        return seq_encs, seq_embs

    def sdeProp(self, geo_encs, seq_encs, target=None):
        local_embs = geo_encs
        condition_embs = seq_encs.detach()
        sde_encs = self.sde.reverse_sde(local_embs, condition_embs, self.diffsize)
        fisher_loss = None
        if target is not None:  # training phase
            t_sampled = np.random.randint(1, self.step_num) / self.step_num
            mean, std = self.sde.marginal_prob(target, t_sampled)
            z = torch.randn_like(target)
            perturbed_data = mean + std.unsqueeze(-1) * z
            score = - self.sde.calc_score(perturbed_data, condition_embs)
            fisher_loss = torch.square(score + z).mean()
        return sde_encs, fisher_loss

    def getTrainLoss(self, batch):
        seq_graph, seqs, label = batch[:3]
        label = torch.LongTensor(label).to(self.device)
        seqs = tuple(seqs)
        poi_embs = self.poi_emb
        poi_embs = self.dropout(poi_embs)
        seq_graph = seq_graph.to(self.device)

        seq_encs, seq_embs = self.seqProp(poi_embs, seq_graph)
        geo_encs, geo_embs = self.geoProp(poi_embs, seqs, seq_encs)
        sde_encs, fisher_loss = self.sdeProp(geo_encs, seq_encs, target=geo_embs[label])
        if self.model_type == 'nu':
            pred_logits = sde_encs @ geo_embs.T
        else:
            pred_logits = seq_encs @ self.poi_emb.T + sde_encs @ geo_embs.T
        return self.ce_criteria(pred_logits, label) + 0.2 * fisher_loss

    def forward(self, seqs, seq_graph):
        seqs = tuple(seqs)
        poi_embs = self.poi_emb
        seq_graph = seq_graph.to(self.device)
        seq_encs, seq_embs = self.seqProp(poi_embs, seq_graph)
        geo_encs, geo_embs = self.geoProp(poi_embs, seqs, seq_encs)
        sde_encs, _ = self.sdeProp(geo_encs, seq_encs)

        if self.model_type == 'nu':
            pred_logits = sde_encs @ geo_embs.T
        else:
            pred_logits = seq_encs @ self.poi_emb.T + sde_encs @ geo_embs.T
        return pred_logits, geo_encs


class SeqEncoder(nn.Module):
    def __init__(self, hid_dim):
        super(SeqEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.encoder = SeqConv(hid_dim)

    def encode(self, embs, seq_graph):
        return self.encoder(embs, seq_graph)


class GeoEncoder(nn.Module):
    def __init__(self, n_poi, hid_dim, geo_graph, args):
        super(GeoEncoder, self).__init__()
        self.n_poi, self.hid_dim = n_poi, hid_dim
        self.gcn_num = args.num_layer

        edge_index, _ = add_self_loops(geo_graph.edge_index)
        dist_vec = torch.cat([geo_graph.edge_attr, torch.zeros((n_poi,)).to(args.device)])
        dist_vec = torch.exp(-(dist_vec ** 2))
        self.geo_graph = Data(edge_index=edge_index, edge_attr=dist_vec)

        self.act = nn.LeakyReLU()
        self.geo_convs = nn.ModuleList()
        for _ in range(self.gcn_num):
            self.geo_convs.append(GeoConv(self.hid_dim, self.hid_dim))

    def encode(self, poi_embs):
        layer_embs = poi_embs
        geo_embs = [layer_embs]
        for conv in self.geo_convs:
            layer_embs = conv(layer_embs, self.geo_graph)
            layer_embs = self.act(layer_embs)
            geo_embs.append(layer_embs)
        geo_embs = torch.stack(geo_embs, dim=1).mean(1)
        return geo_embs
