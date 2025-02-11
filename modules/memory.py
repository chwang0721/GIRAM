import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryModule:
    def __init__(self, user_count, value_dim, memory_size, device, alpha, key_dim=48, similarity_threshold=0.95):
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.similarity_threshold = similarity_threshold
        self.device = device
        self.alpha = alpha

        self.user_memory = {
            user_id: {
                "keys": torch.zeros(memory_size, key_dim, device=device),
                "values": [None for _ in range(memory_size)],
                "timestamps": torch.zeros((memory_size,), device=device)
            }
            for user_id in range(user_count)
        }

    def find_best_match(self, keys, query_keys):
        similarities = F.cosine_similarity(query_keys.unsqueeze(1), keys.unsqueeze(0), dim=2)
        max_similarities, idxs = similarities.max(dim=1)  # shape: (num_queries,)
        return idxs, max_similarities

    def sparse_top_k(self, dense_values, k=50):
        top_k_values, top_k_indices = torch.topk(dense_values, k=k, sorted=False)
        mask = top_k_values > 0
        top_k_values = top_k_values[mask]
        top_k_indices = top_k_indices[mask]
        sparse_tensor = torch.sparse_coo_tensor(
            indices=top_k_indices.unsqueeze(0),
            values=top_k_values,
            size=(self.value_dim,),
            device=self.device
        )
        return sparse_tensor

    def update_user_memory(self, user_id, query_keys, query_values, current_times, user_sim):
        user_memory = self.user_memory[user_id]
        keys, values_list, timestamps = user_memory["keys"], user_memory["values"], user_memory["timestamps"]

        dynamic_alpha = self.alpha + 0.5 * (user_sim - 0.5)
        dynamic_threshold = self.similarity_threshold
        idxs, max_similarities = self.find_best_match(keys, query_keys)  # shape: (num_queries,)

        for idx, max_similarity, key, value, current_time in zip(idxs, max_similarities,
                                                                 query_keys, query_values, current_times):
            sparse_value = self.sparse_top_k(value, k=50)

            if max_similarity >= dynamic_threshold:
                keys[idx] = (1 - dynamic_alpha) * keys[idx] + dynamic_alpha * key

                if values_list[idx] is None:
                    values_list[idx] = sparse_value
                else:
                    existing_value = values_list[idx].to_dense()
                    updated_value = (1 - dynamic_alpha) * existing_value + dynamic_alpha * value
                    values_list[idx] = self.sparse_top_k(updated_value, k=50)

                timestamps[idx] = current_time
            else:
                empty_idx = (keys.norm(dim=1) == 0).nonzero(as_tuple=True)[0]
                if len(empty_idx) > 0:
                    idx_to_add = empty_idx[0].item()
                else:
                    idx_to_add = timestamps.argmin().item()
                keys[idx_to_add] = key
                values_list[idx_to_add] = sparse_value
                timestamps[idx_to_add] = current_time

    def retrieve_memory(self, user_id, query_keys):
        user_memory = self.user_memory[user_id]
        keys, values_list = user_memory["keys"], user_memory["values"]
        mask = keys.norm(dim=1) != 0
        keys = keys[mask]
        values_list = [values_list[i] for i in range(len(values_list)) if mask[i]]

        if len(keys) == 0:
            return torch.zeros(self.value_dim, device=self.device)

        all_similarities = F.cosine_similarity(query_keys.unsqueeze(1), keys.unsqueeze(0), dim=2)
        sorted_indices = torch.argsort(all_similarities, dim=1, descending=True)

        rank_indices = torch.arange(1, keys.size(0) + 1, device=self.device).float()
        rrf_weights = 1 / (rank_indices + 50)
        rrf_scores = torch.zeros(len(keys), device=self.device)

        for query_idx in range(query_keys.size(0)):
            rrf_scores += rrf_weights.scatter(0, sorted_indices[query_idx], rrf_weights)

        top_k_rrf_scores, top_k_indices = torch.topk(rrf_scores, min(20, len(rrf_scores)))
        attn_weights = F.softmax(top_k_rrf_scores, dim=0)

        top_k_values = torch.stack([
            values_list[idx].to_dense() if values_list[idx] is not None else torch.zeros(self.value_dim, device=self.device)
            for idx in top_k_indices
        ])
        final_personal_interest = torch.matmul(attn_weights, top_k_values)

        return final_personal_interest


class SequenceModel(nn.Module):
    def __init__(self, input_dim):
        super(SequenceModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, input_dim, batch_first=True)
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = self.linear(x)
        x, _ = self.encoder(x)
        return x


class TimeKeyEncoder(nn.Module):
    def __init__(self, embedding_dim=5, max_hour=24, max_weekday=7):
        super(TimeKeyEncoder, self).__init__()
        self.hour_embedding = nn.Embedding(max_hour, embedding_dim)
        self.weekday_embedding = nn.Embedding(max_weekday, embedding_dim)

    def discrete_time_embedding(self, hour, weekday):
        hour_emb = self.hour_embedding(hour)
        weekday_emb = self.weekday_embedding(weekday)
        return torch.cat([hour_emb, weekday_emb], dim=-1)

    def sine_cosine_encoding(self, norm_time, frequencies=[1, 2, 4]):
        embeddings = []
        for freq in frequencies:
            sine = torch.sin(2 * torch.pi * freq * norm_time)
            cosine = torch.cos(2 * torch.pi * freq * norm_time)
            embeddings.append(torch.stack([sine, cosine], dim=-1))
        return torch.cat(embeddings, dim=-1)

    def forward(self, hour, weekday, norm_time):
        discrete_emb = self.discrete_time_embedding(hour, weekday)
        sine_cosine_emb = self.sine_cosine_encoding(norm_time)
        return torch.cat([discrete_emb, sine_cosine_emb], dim=-1)


class GeoKeyEncoder(nn.Module):
    def __init__(self, geo_range, region_num):
        super(GeoKeyEncoder, self).__init__()
        self.coord_embedding = nn.Linear(2, 6)
        self.lat_min, self.lat_max = geo_range[0]
        self.lon_min, self.lon_max = geo_range[1]
        self.region_embedding = nn.Embedding(region_num, 10)

    def forward(self, location, region_id):
        # Normalize the location
        location[:, :, 0] = (location[:, :, 0] - self.lat_min) / (self.lat_max - self.lat_min)
        location[:, :, 1] = (location[:, :, 1] - self.lon_min) / (self.lon_max - self.lon_min)
        coord_emb = self.coord_embedding(location)
        region_emb = self.region_embedding(region_id)
        return torch.cat([coord_emb, region_emb], dim=-1)


class CategoryKeyEncoder(nn.Module):
    def __init__(self, main_cat_num, cat_num, main_category_dim=4, sub_category_dim=12):
        super(CategoryKeyEncoder, self).__init__()
        self.main_category_embedding = nn.Embedding(main_cat_num, main_category_dim)
        self.sub_category_embedding = nn.Embedding(cat_num, sub_category_dim)

    def forward(self, main_category_id, sub_category_id):
        main_emb = self.main_category_embedding(main_category_id)
        sub_emb = self.sub_category_embedding(sub_category_id)
        return torch.cat([main_emb, sub_emb], dim=-1)


class KeyEncoder(nn.Module):
    def __init__(self, geo_range, main_cat_num, cat_num, region_num, device):
        super(KeyEncoder, self).__init__()
        self.time_encoder = TimeKeyEncoder()
        self.geo_encoder = GeoKeyEncoder(geo_range, region_num)
        self.cat_encoder = CategoryKeyEncoder(main_cat_num, cat_num)
        self.sequence_encoder = SequenceModel(48)
        self.device = device

    def forward(self, hour_inputs, weekday_inputs, norm_time_inputs, location_inputs, main_cat_inputs, cat_inputs, region_inputs):
        time_key = self.time_encoder(hour_inputs.to(self.device), weekday_inputs.to(self.device),
                                                    norm_time_inputs.to(self.device))
        geo_key = self.geo_encoder(location_inputs.to(self.device), region_inputs.to(self.device))
        cat_key = self.cat_encoder(main_cat_inputs.to(self.device), cat_inputs.to(self.device))
        key = torch.cat([time_key, geo_key, cat_key], dim=-1)
        key = self.sequence_encoder(key)
        return key
