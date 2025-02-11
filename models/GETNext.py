import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, TransformerEncoder, TransformerEncoderLayer


class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout)
        x = self.gcn[-1](x, adj)

        return x


class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), -1))
        x = self.leaky_relu(x)
        return x


def t2v(tau, f, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat, x


class GETNext(nn.Module):
    def __init__(self, in_channels, poi_num, user_num, cat_num, args, model_type='u'):
        super(GETNext, self).__init__()
        self.poi_embed_model = GCN(ninput=in_channels,
                                   nhid=args.gcn_nhid,
                                   noutput=args.poi_embed_dim,
                                   dropout=args.gcn_dropout)

        self.node_attn_model = NodeAttnMap(in_features=in_channels,
                                           nhid=args.node_attn_nhid, use_mask=True)

        self.user_embed_model = UserEmbeddings(user_num, args.user_embed_dim)
        if model_type == 'nu':
            nn.init.normal_(self.user_embed_model.user_embedding.weight, mean=0.0, std=0.01)
            self.user_embed_model.user_embedding.weight.requires_grad = False

        self.time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)

        self.cat_embed_model = CategoryEmbeddings(cat_num, args.cat_embed_dim)

        self.embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
        self.embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)

        seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
        self.seq_model = TransformerModel(poi_num,
                                          cat_num,
                                          seq_input_embed,
                                          args.transformer_nhead,
                                          args.transformer_nhid,
                                          args.transformer_nlayers,
                                          dropout=args.transformer_dropout)
        self.device = args.device

    def traj_to_embeddings(self, user_idx, input_seq, input_seq_time, input_seq_cat, poi_embeddings):
        input = torch.LongTensor(user_idx).to(self.device)
        user_embedding = self.user_embed_model(input)
        user_seq_embedding = user_embedding.unsqueeze(1).expand(-1, 100, -1)

        poi_seq_embedding = poi_embeddings[input_seq]

        time_seq_embedding = self.time_embed_model(input_seq_time.to(self.device).unsqueeze(-1))
        cat_seq_embedding = self.cat_embed_model(input_seq_cat.to(self.device))

        fused_embedding1 = self.embed_fuse_model1(user_seq_embedding, poi_seq_embedding)
        fused_embedding2 = self.embed_fuse_model2(time_seq_embedding, cat_seq_embedding)

        input_seq_embed = torch.cat((fused_embedding1, fused_embedding2), dim=-1)
        return input_seq_embed

    def get_seq_embeds(self, users, poi_inputs, norm_time_inputs, cat_inputs, poi_embeddings):
        input_seq_embed = self.traj_to_embeddings(users, poi_inputs, norm_time_inputs, cat_inputs, poi_embeddings)
        return input_seq_embed

    def adjust_pred(self, y_pred_poi, batch_input_seqs, X, A):
        attn_map = self.node_attn_model(X, A)
        attn_values = attn_map[batch_input_seqs]
        y_pred_poi_adjusted = attn_values + y_pred_poi

        return y_pred_poi_adjusted

    def forward(self, X, A, users, poi_inputs, norm_time_inputs, cat_inputs):
        poi_embeddings = self.poi_embed_model(X, A)
        poi_inputs = poi_inputs.to(self.device)
        seq_embeds = self.get_seq_embeds(users, poi_inputs, norm_time_inputs, cat_inputs, poi_embeddings)

        src_mask = self.seq_model.generate_square_subsequent_mask(poi_inputs.size(1)).to(self.device)
        poi_outputs, norm_time_outputs, cat_outputs, reps = self.seq_model(seq_embeds, src_mask)
        poi_outputs = self.adjust_pred(poi_outputs, poi_inputs, X, A)

        return poi_outputs, norm_time_outputs, cat_outputs, reps
