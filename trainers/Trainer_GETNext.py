import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.GETNext import GETNext
from modules.KeyVAE import KeyVAE
from modules.memory import MemoryModule, KeyEncoder
from trainers.utils import make_mask, top_k_acc, mean_reciprocal_rank


class Trainer_GETNext:
    def __init__(self, args, poi_num, user_num, cat_num, main_cat_num, memory_size, A, X, in_channels, geo_range,
                 region_num):
        self.device = args.device
        self.X = X
        self.A = A

        self.model = GETNext(in_channels, poi_num, user_num, cat_num, args).to(self.device)
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

        if args.mode == "pretrain" or args.mode == 'memory':
            self.model_nu = GETNext(in_channels, poi_num, user_num, cat_num, args, 'nu').to(self.device)
            self.optimizer_nu = optim.Adam(params=self.model_nu.parameters(),
                                           lr=args.lr,
                                           weight_decay=args.weight_decay)

        self.criterion_poi = nn.CrossEntropyLoss(reduction='none')
        self.criterion_cat = nn.CrossEntropyLoss(reduction='none')
        self.criterion_time = nn.MSELoss(reduction='none')
        self.time_loss_weight = args.time_loss_weight
        self.kd_loss = nn.KLDivLoss(reduction='none')
        self.beta = args.beta

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', factor=args.lr_scheduler_factor)

        if args.mode == 'memory':
            self.key_encoder = KeyEncoder(geo_range, main_cat_num, cat_num, region_num, self.device).to(self.device)
            self.memory = MemoryModule(user_num, poi_num, memory_size, self.device, args.alpha)

        self.n_samples = args.n_samples

    def get_loss(self, poi_outputs, norm_time_outputs, cat_outputs, lengths, poi_labels, norm_time_labels, cat_labels):
        poi_labels = poi_labels.to(self.device)
        norm_time_labels = norm_time_labels.to(self.device)
        cat_labels = cat_labels.to(self.device)
        mask = make_mask(poi_labels, lengths)
        loss_poi = (self.criterion_poi(poi_outputs.transpose(1, 2), poi_labels) * mask).sum() / mask.sum()
        loss_time = (self.criterion_time(torch.squeeze(norm_time_outputs), norm_time_labels)).sum() / mask.sum()
        loss_cat = (self.criterion_cat(cat_outputs.transpose(1, 2), cat_labels) * mask).sum() / mask.sum()
        loss = loss_poi + loss_time * self.time_loss_weight + loss_cat
        return loss

    def update_batch_memory(self, batch, user_sim):
        (users, poi_inputs, timestamp_inputs, location_inputs, hour_inputs, weekday_inputs, norm_time_inputs,
         main_cat_inputs, cat_inputs, region_inputs, poi_labels, _, _, lengths) = batch

        keys = self.key_encoder(hour_inputs, weekday_inputs, norm_time_inputs,
                                location_inputs, main_cat_inputs, cat_inputs, region_inputs)

        poi_outputs, _, _, _ = self.model(self.X, self.A, users, poi_inputs, norm_time_inputs, cat_inputs)
        poi_outputs = F.softmax(poi_outputs, dim=-1)

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
                (users, poi_inputs, _, _, _, _, norm_time_inputs, _, cat_inputs,
                 poi_labels, _, norm_time_labels, cat_labels, lengths) = batch

                poi_outputs = self.model(self.X, self.A, users, poi_inputs, norm_time_inputs, cat_inputs)[0]
                poi_outputs_nu = self.model_nu(self.X, self.A, users, poi_inputs, norm_time_inputs, cat_inputs)[0]

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
            (users, poi_inputs, _, _, _, _, norm_time_inputs, _, cat_inputs, _,
             poi_labels, norm_time_labels, cat_labels, lengths) = batch
            poi_outputs, norm_time_outputs, cat_outputs, _ = self.model(self.X, self.A, users, poi_inputs,
                                                                        norm_time_inputs, cat_inputs)

            loss = self.get_loss(poi_outputs, norm_time_outputs, cat_outputs, lengths, poi_labels.to(self.device),
                                 norm_time_labels.to(self.device), cat_labels.to(self.device))

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader)

    def train_nu(self, train_loader):
        self.model_nu.train()

        train_loss = 0
        for batch in train_loader:
            (users, poi_inputs, _, _, _, _, norm_time_inputs, _, cat_inputs, _,
             poi_labels, norm_time_labels, cat_labels, lengths) = batch
            poi_outputs, norm_time_outputs, cat_outputs, _ = self.model_nu(self.X, self.A, users, poi_inputs,
                                                                           norm_time_inputs, cat_inputs)

            loss = self.get_loss(poi_outputs, norm_time_outputs, cat_outputs, lengths, poi_labels.to(self.device),
                                 norm_time_labels.to(self.device), cat_labels.to(self.device))

            self.optimizer_nu.zero_grad()
            loss.backward(retain_graph=True)
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
                users, poi_inputs, _, _, _, _, norm_time_inputs, _, cat_inputs, _, poi_labels, _, _, lengths = batch

                poi_labels = poi_labels.to(self.device)

                poi_outputs = self.model(self.X, self.A, users, poi_inputs, norm_time_inputs, cat_inputs)[0]
                poi_outputs_nu = self.model_nu(self.X, self.A, users, poi_inputs, norm_time_inputs, cat_inputs)[0]

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

            return acc_5 / cnt, acc_10 / cnt, acc_20 / cnt, mrr / cnt, acc_5_nu / cnt, acc_10_nu / cnt, acc_20_nu / cnt, mrr_nu / cnt

    def test(self, test_loader):
        self.model.eval()
        acc_5, acc_10, acc_20, mrr = 0, 0, 0, 0
        cnt = 0
        with torch.no_grad():
            for batch in test_loader:
                users, poi_inputs, _, _, _, _, norm_time_inputs, _, cat_inputs, _, poi_labels, _, _, lengths = batch
                poi_labels = poi_labels.to(self.device)

                poi_outputs, _, _, _ = self.model(self.X, self.A, users, poi_inputs, norm_time_inputs, cat_inputs)

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

        for epoch in range(1):
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

                poi_outputs, _, _, _ = self.model(self.X, self.A, users, poi_inputs, norm_time_inputs, cat_inputs)
                poi_outputs = F.softmax(poi_outputs, dim=-1)

                batch_indices = torch.arange(keys.size(0))
                last_keys = keys[batch_indices, lengths - 1]  # [batch_size, key_dim]
                short_term_interest = poi_outputs[batch_indices, lengths - 1]  # [batch_size, poi_num]
                labels = poi_labels[batch_indices, lengths - 1].to(self.device)

                interest = torch.zeros_like(short_term_interest)

                for i, user in enumerate(users):
                    user = int(user)
                    user_sim = user_similarity_dict.get(user, 0.5)
                    query_keys = self.KeyVAE.generate(last_keys[i].unsqueeze(0))
                    query_keys = torch.cat([last_keys[i].unsqueeze(0), query_keys.squeeze(0)], dim=0)
                    long_term_interest = self.memory.retrieve_memory(user, query_keys)
                    beta = self.beta + (user_sim - 0.5) * 0.5
                    interest[i] = long_term_interest * beta + short_term_interest[i] * (1 - beta)

                outputs = interest
                acc_5 += top_k_acc(labels, outputs, 5)
                acc_10 += top_k_acc(labels, outputs, 10)
                acc_20 += top_k_acc(labels, outputs, 20)
                mrr += mean_reciprocal_rank(labels, outputs)
                cnt += len(users)

            return acc_5 / cnt, acc_10 / cnt, acc_20 / cnt, mrr / cnt
