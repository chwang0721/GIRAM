import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class Flashback(nn.Module):
    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, device, model_type='u'):
        super().__init__()
        self.input_size = input_size
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t
        self.f_s = f_s

        self.encoder = nn.Embedding(input_size, hidden_size)  # location embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size)  # user embedding
        if model_type == 'nu':
            nn.init.normal_(self.user_encoder.weight, 0, 0.1)
            self.user_encoder.weight.requires_grad = False
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)  # RNN unit
        self.fc = nn.Linear(2 * hidden_size, input_size)
        self.device = device

    def forward(self, poi_inputs, lengths, timestamp_inputs, location_inputs, active_user):
        x = poi_inputs.to(self.device)
        t = timestamp_inputs.to(self.device)
        s = location_inputs.to(self.device)
        active_user = active_user.to(self.device)

        user_len, seq_len = x.size()  # x: (batch_size, seq_len)
        x_emb = self.encoder(x)
        x_emb = pack_padded_sequence(x_emb, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(x_emb)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=seq_len)

        out = out.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        t = t.transpose(0, 1)  # (seq_len, batch_size)
        s = s.transpose(0, 1)  # (seq_len, batch_size, 2)

        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
        for i in range(seq_len):
            dist_t = t[i].unsqueeze(0) - t[:i + 1]
            dist_t = torch.clamp_min(dist_t, 0)
            dist_s = torch.norm(s[i].unsqueeze(0) - s[:i + 1], dim=-1)
            a_j = self.f_t(dist_t, user_len).unsqueeze(-1)
            b_j = self.f_s(dist_s, user_len).unsqueeze(-1)
            # Compute the weights
            w_j = a_j * b_j + 1e-10
            sum_w = w_j.sum(dim=0)
            out_w[i] = (w_j * out[:i + 1]).sum(dim=0)
            out_w[i] /= torch.clamp(sum_w, min=1e-10)

        p_u = self.user_encoder(active_user).repeat(seq_len, 1, 1)
        out_pu = torch.cat([out_w, p_u], dim=-1)  # (seq_len, batch_size, 2 * hidden_size)
        y_linear = self.fc(out_pu).transpose(0, 1)  # (batch_size, seq_len, input_size)
        y_linear = F.softmax(y_linear, dim=-1)

        return y_linear, out_pu.transpose(0, 1)
