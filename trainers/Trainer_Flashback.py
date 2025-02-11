import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.Flashback import Flashback
from modules.KeyVAE import KeyVAE
from modules.memory import KeyEncoder, MemoryModule
from trainers.utils import make_mask, top_k_acc, mean_reciprocal_rank


class Trainer_Flashback:
    def __init__(self, args, lambda_t, lambda_s, poi_num, user_num, main_cat_num, cat_num,
                 memory_size, geo_range, region_num):
        self.device = args.device

        # Time and space decay functions
        f_t = lambda delta_t, _: ((torch.cos(delta_t * 2 * torch.pi / 86400) + 1) / 2) * \
                                 torch.exp(-(delta_t / 86400 * lambda_t))
        f_s = lambda delta_s, _: torch.exp(-(delta_s * lambda_s))

        # Model and optimizer
        self.model = Flashback(poi_num, user_num, args.hidden_size, f_t, f_s, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.model_nu = Flashback(poi_num, user_num, args.hidden_size, f_t, f_s, self.device, 'nu').to(self.device)
        self.optimizer_nu = optim.Adam(self.model_nu.parameters(), lr=args.lr)

        # Loss functions
        self.loss_fn = nn.NLLLoss(reduction="none")
        self.kd_loss = nn.KLDivLoss(reduction="none")

        self.key_encoder = KeyEncoder(geo_range, main_cat_num, cat_num, region_num, self.device).to(self.device)
        self.memory = MemoryModule(user_num, poi_num, memory_size, self.device, args.alpha)

        self.beta = args.beta
        self.n_samples = args.n_samples
        self.poi_num = poi_num

    def update_batch_memory(self, batch, user_sim):
        (users, poi_inputs, timestamp_inputs, location_inputs, hour_inputs, weekday_inputs, norm_time_inputs,
         main_cat_inputs, cat_inputs, region_inputs, poi_labels, _, _, lengths) = batch

        keys = self.key_encoder(hour_inputs, weekday_inputs, norm_time_inputs,
                                location_inputs, main_cat_inputs, cat_inputs, region_inputs)
        poi_outputs, _ = self.model(poi_inputs, lengths, timestamp_inputs, location_inputs, users)

        for i, user in enumerate(users):
            u = int(user)

            self.memory.update_user_memory(
                user_id=u,
                query_keys=keys[i][:lengths[i]],
                query_values=poi_outputs[i][:lengths[i]],
                current_times=timestamp_inputs[i][:lengths[i]],
                user_sim=user_sim.get(u, 0.5)
            )

    def update_memory(self, train_loader, user_sim):
        self.model.eval()
        with torch.no_grad():
            for batch in train_loader:
                self.update_batch_memory(batch, user_sim)

    def get_similarities(self, test_loader):
        self.model.eval()
        self.model_nu.eval()

        all_similarities = []
        all_users = []

        with torch.no_grad():
            for batch in test_loader:
                users, poi_inputs, timestamp_inputs, location_inputs, _, _, _, _, _, _, _, _, _, lengths = batch

                poi_outputs, _ = self.model(poi_inputs, lengths, timestamp_inputs, location_inputs, users)
                poi_outputs_nu, _ = self.model_nu(poi_inputs, lengths, timestamp_inputs, location_inputs, users)

                for i, length in enumerate(lengths):
                    output = poi_outputs[i, :length]
                    output_nu = poi_outputs_nu[i, :length]
                    similarity = F.cosine_similarity(output, output_nu, dim=-1)
                    all_similarities.append(similarity)
                    all_users.append(users[i].repeat(length))

        all_similarities = torch.cat(all_similarities, dim=0)
        all_users = torch.cat(all_users, dim=0)

        user_similarities = {}
        for user in all_users.unique():
            mask = all_users == user
            user_similarities[user.item()] = all_similarities[mask].mean().item()
        return user_similarities

    def train(self, train_loader):
        self.model.train()
        train_loss = 0

        for batch in train_loader:
            users, poi_inputs, timestamp_inputs, location_inputs, _, _, _, _, _, _, poi_labels, _, _, lengths = batch
            poi_labels = poi_labels.to(self.device)

            poi_outputs, _ = self.model(poi_inputs, lengths, timestamp_inputs, location_inputs, users)
            loss = self.loss_fn(torch.log(poi_outputs.transpose(1, 2)), poi_labels)
            mask = make_mask(poi_labels, lengths)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        return train_loss / len(train_loader)

    def train_nu(self, train_loader):
        self.model_nu.train()
        train_loss = 0

        for batch in train_loader:
            users, poi_inputs, timestamp_inputs, location_inputs, _, _, _, _, _, _, poi_labels, _, _, lengths = batch
            poi_labels = poi_labels.to(self.device)

            poi_outputs, _ = self.model_nu(poi_inputs, lengths, timestamp_inputs, location_inputs, users)
            loss = self.loss_fn(torch.log(poi_outputs.transpose(1, 2)), poi_labels)
            mask = make_mask(poi_labels, lengths)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer_nu.zero_grad()
            loss.backward()
            self.optimizer_nu.step()
            train_loss += loss.item()

        return train_loss / len(train_loader)

    def test_pretrain(self, test_loader):
        acc_5, acc_10, acc_20, mrr = 0, 0, 0, 0
        acc_5_nu, acc_10_nu, acc_20_nu, mrr_nu = 0, 0, 0, 0
        self.model.eval()
        self.model_nu.eval()
        cnt = 0

        with torch.no_grad():
            for batch in test_loader:
                users, poi_inputs, timestamp_inputs, location_inputs, _, _, _, _, _, _, poi_labels, _, _, lengths = batch

                poi_labels = poi_labels.to(self.device)

                poi_outputs, _ = self.model(poi_inputs, lengths, timestamp_inputs, location_inputs, users)
                poi_outputs_nu, _ = self.model_nu(poi_inputs, lengths, timestamp_inputs, location_inputs, users)

                batch_indices = torch.arange(len(users))
                labels = poi_labels[batch_indices, lengths - 1]
                outputs = poi_outputs[batch_indices, lengths - 1]
                outputs_nu = poi_outputs_nu[batch_indices, lengths - 1]

                acc_5 += top_k_acc(labels, outputs, 5)
                acc_10 += top_k_acc(labels, outputs, 10)
                acc_20 += top_k_acc(labels, outputs, 20)
                mrr += mean_reciprocal_rank(labels, outputs)

                acc_5_nu += top_k_acc(labels, outputs_nu, 5)
                acc_10_nu += top_k_acc(labels, outputs_nu, 10)
                acc_20_nu += top_k_acc(labels, outputs_nu, 20)
                mrr_nu += mean_reciprocal_rank(labels, outputs_nu)
                cnt += len(users)

            return acc_5 / cnt, acc_10 / cnt, acc_20 / cnt, mrr / cnt, \
                   acc_5_nu / cnt, acc_10_nu / cnt, acc_20_nu / cnt, mrr_nu / cnt

    def test(self, test_loader):
        acc_5, acc_10, acc_20, mrr = 0, 0, 0, 0
        self.model.eval()
        cnt = 0

        with torch.no_grad():
            for batch in test_loader:
                users, poi_inputs, timestamp_inputs, location_inputs, _, _, _, _, _, _, poi_labels, _, _, lengths = batch
                poi_labels = poi_labels.to(self.device)
                poi_outputs, _ = self.model(poi_inputs, lengths, timestamp_inputs, location_inputs, users)

                batch_indices = torch.arange(len(users))
                labels = poi_labels[batch_indices, lengths - 1]
                outputs = poi_outputs[batch_indices, lengths - 1]

                acc_5 += top_k_acc(labels, outputs, 5)
                acc_10 += top_k_acc(labels, outputs, 10)
                acc_20 += top_k_acc(labels, outputs, 20)
                mrr += mean_reciprocal_rank(labels, outputs)
                cnt += len(users)

            return acc_5 / cnt, acc_10 / cnt, acc_20 / cnt, mrr / cnt

    def train_vae(self):
        self.KeyVAE = KeyVAE(48, self.n_samples).to(self.device)
        optimizer = torch.optim.AdamW(self.KeyVAE.parameters(), lr=0.005)

        all_keys = self.memory.user_memory.values()
        all_keys = torch.cat([keys["keys"] for keys in all_keys], dim=0)
        mask = all_keys.norm(dim=1) != 0
        all_keys = all_keys[mask]

        train_loader = torch.utils.data.DataLoader(all_keys, batch_size=512, shuffle=True)

        for epoch in range(50):
            for batch in train_loader:
                optimizer.zero_grad()
                batch = batch.to(self.device)
                generated_keys, mu, logvar = self.KeyVAE(batch)
                loss = self.KeyVAE.KeyVAE_loss(generated_keys, batch, mu, logvar)
                loss.backward()
                optimizer.step()

    def test_memory(self, test_loader, user_similarity_dict):
        acc_5, acc_10, acc_20, mrr = 0, 0, 0, 0
        self.model.eval()
        cnt = 0

        with torch.no_grad():
            for batch in test_loader:
                (users, poi_inputs, timestamp_inputs, location_inputs, hour_inputs, weekday_inputs, norm_time_inputs,
                 main_cat_inputs, cat_inputs, region_inputs, poi_labels, _, _, lengths) = batch

                keys = self.key_encoder(hour_inputs, weekday_inputs, norm_time_inputs, location_inputs,
                                        main_cat_inputs, cat_inputs, region_inputs)

                poi_outputs, _ = self.model(poi_inputs, lengths, timestamp_inputs, location_inputs, users)

                batch_indices = torch.arange(keys.size(0))
                last_keys = keys[batch_indices, lengths - 1]  # [batch_size, key_dim]
                short_term_interest = poi_outputs[batch_indices, lengths - 1]  # [batch_size, poi_num]
                labels = poi_labels[batch_indices, lengths - 1].to(self.device)

                interest = torch.zeros_like(short_term_interest)
                batch_query_keys = self.KeyVAE.generate(last_keys)
                batch_query_keys = torch.cat([last_keys.unsqueeze(1), batch_query_keys], dim=1)

                for i, user in enumerate(users):
                    user = int(user)
                    user_sim = user_similarity_dict.get(user, 0.5)
                    long_term_interest = self.memory.retrieve_memory(user, batch_query_keys[i])
                    beta = self.beta + (user_sim - 0.5) * 0.5
                    interest[i] = long_term_interest * (1 - beta) + short_term_interest[i] * beta

                outputs = interest
                acc_5 += top_k_acc(labels, outputs, 5)
                acc_10 += top_k_acc(labels, outputs, 10)
                acc_20 += top_k_acc(labels, outputs, 20)
                mrr += mean_reciprocal_rank(labels, outputs)
                cnt += len(users)

            return acc_5 / cnt, acc_10 / cnt, acc_20 / cnt, mrr / cnt
