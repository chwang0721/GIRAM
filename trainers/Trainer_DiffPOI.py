import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.DiffPOI import DiffPOI
from modules.KeyVAE import KeyVAE
from modules.memory import KeyEncoder, MemoryModule
from trainers.utils import top_k_acc, mean_reciprocal_rank


class Trainer_DiffPOI:
    def __init__(self, user_num, poi_num, geo_graph, args, main_cat_num, cat_num,
                 memory_size, geo_range, region_num):
        self.device = args.device
        self.model = DiffPOI(user_num, poi_num, geo_graph, args).to(args.device)
        self.opt = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.decay)
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')

        if args.mode == "pretrain" or args.mode == "memory":
            self.model_nu = DiffPOI(user_num, poi_num, geo_graph, args, 'nu').to(args.device)
            self.opt_nu = optim.AdamW(self.model_nu.parameters(), lr=args.lr, weight_decay=args.decay)

        if args.mode == "memory":
            self.key_encoder = KeyEncoder(geo_range, main_cat_num, cat_num, region_num, self.device).to(self.device)
            self.memory = MemoryModule(user_num, poi_num, memory_size, self.device, args.alpha)
        self.beta = args.b
        self.n_samples = args.n_samples

    def train(self, loader):
        self.model.train()
        train_loss = 0
        for batch in loader:
            loss = self.model.getTrainLoss(batch)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            train_loss += loss.item()
        return train_loss / len(loader)

    def train_nu(self, loader):
        self.model_nu.train()
        train_loss = 0
        for batch in loader:
            loss = self.model_nu.getTrainLoss(batch)
            self.opt_nu.zero_grad()
            loss.backward()
            self.opt_nu.step()
            train_loss += loss.item()
        return train_loss / len(loader)

    def update_batch_memory(self, batch, user_sim):
        (seq_graph, seqs, poi_labels, timestamp_inputs, hour_inputs, weekday_inputs, norm_time_inputs, location_inputs,
         main_cat_inputs, cat_inputs, region_inputs, lengths, users, exclude_masks) = batch

        keys = self.key_encoder(hour_inputs, weekday_inputs, norm_time_inputs,
                                location_inputs, main_cat_inputs, cat_inputs, region_inputs)

        poi_outputs, _ = self.model(seqs, seq_graph)
        batch_indices = torch.arange(keys.size(0))
        keys = keys[batch_indices, lengths - 1]  # [batch_size key_dim]
        timestamp_inputs = timestamp_inputs[batch_indices, lengths - 1]

        for i, user in enumerate(users):
            u = int(user)

            self.memory.update_user_memory(
                user_id=u,
                query_keys=keys[i].unsqueeze(0),
                query_values=poi_outputs[i].unsqueeze(0),
                current_times=timestamp_inputs[i].unsqueeze(0),
                user_sim=user_sim.get(u, 0.5)
            )

    def update_memory(self, train_loader, user_sim):
        self.model.eval()
        with torch.no_grad():
            for batch in train_loader:
                self.update_batch_memory(batch, user_sim)

    def get_similarities(self, loader):
        self.model.eval()
        self.model_nu.eval()

        all_similarities = []
        all_users = []

        with torch.no_grad():
            for batch in loader:
                seq_graph, seqs = batch[:2]
                users = batch[-2]

                outputs, _ = self.model(seqs, seq_graph)
                outputs_nu, _ = self.model_nu(seqs, seq_graph)

                for i in range(len(outputs)):
                    output = outputs[i]
                    output_nu = outputs_nu[i]
                    similarity = F.cosine_similarity(output, output_nu, dim=0)
                    all_similarities.append(similarity)
                    all_users.append(users[i])

        all_similarities = torch.Tensor(all_similarities).to(self.device)
        all_users = torch.LongTensor(all_users).to(self.device)

        user_similarities = {}
        for user in all_users.unique():
            mask = all_users == user
            user_similarities[user.item()] = all_similarities[mask].mean().item()
        return user_similarities

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

    def test_pretrain(self, loader, test_len):
        self.model.eval()
        self.model_nu.eval()
        acc_5, acc_10, acc_20, mrr = 0, 0, 0, 0
        acc_5_nu, acc_10_nu, acc_20_nu, mrr_nu = 0, 0, 0, 0
        with torch.no_grad():
            for batch in loader:
                seq_graph, seqs, labels = batch[:3]
                exclude_masks = torch.stack(batch[-1]).to(self.device)
                labels = labels.to(self.device)
                outputs, _ = self.model(seqs, seq_graph)
                outputs_nu, _ = self.model_nu(seqs, seq_graph)
                outputs[exclude_masks] = -1e10
                outputs_nu[exclude_masks] = -1e10

                acc_5 += top_k_acc(labels, outputs, 5)
                acc_10 += top_k_acc(labels, outputs, 10)
                acc_20 += top_k_acc(labels, outputs, 20)
                mrr += mean_reciprocal_rank(labels, outputs)

                acc_5_nu += top_k_acc(labels, outputs_nu, 5)
                acc_10_nu += top_k_acc(labels, outputs_nu, 10)
                acc_20_nu += top_k_acc(labels, outputs_nu, 20)
                mrr_nu += mean_reciprocal_rank(labels, outputs_nu)

        return acc_5 / test_len, acc_10 / test_len, acc_20 / test_len, mrr / test_len, \
               acc_5_nu / test_len, acc_10_nu / test_len, acc_20_nu / test_len, mrr_nu / test_len

    def test(self, loader, test_len):
        self.model.eval()
        acc_5, acc_10, acc_20, mrr = 0, 0, 0, 0
        with torch.no_grad():
            for batch in loader:
                seq_graph, seqs, labels = batch[:3]
                exclude_masks = torch.stack(batch[-1]).to(self.device)
                labels = labels.to(self.device)
                outputs, _ = self.model(seqs, seq_graph)
                outputs[exclude_masks] = -1e10

                acc_5 += top_k_acc(labels, outputs, 5)
                acc_10 += top_k_acc(labels, outputs, 10)
                acc_20 += top_k_acc(labels, outputs, 20)
                mrr += mean_reciprocal_rank(labels, outputs)

        return acc_5 / test_len, acc_10 / test_len, acc_20 / test_len, mrr / test_len

    def test_memory(self, test_loader, user_similarity_dict):
        acc_5, acc_10, acc_20, mrr = 0, 0, 0, 0
        self.model.eval()
        cnt = 0

        with torch.no_grad():
            for batch in test_loader:
                (seq_graph, seqs, poi_labels, timestamp_inputs, hour_inputs, weekday_inputs, norm_time_inputs,
                 location_inputs, main_cat_inputs, cat_inputs, region_inputs, lengths, users, _) = batch
                exclude_masks = torch.stack(batch[-1]).to(self.device)

                keys = self.key_encoder(hour_inputs, weekday_inputs, norm_time_inputs, location_inputs,
                                        main_cat_inputs, cat_inputs, region_inputs)
                poi_outputs, _ = self.model(seqs, seq_graph)

                batch_indices = torch.arange(keys.size(0))
                last_keys = keys[batch_indices, lengths - 1]  # [batch_size, key_dim]
                short_term_interest = poi_outputs  # [batch_size, poi_num]

                interest = torch.zeros_like(short_term_interest)
                labels = poi_labels.to(self.device)

                for i, user in enumerate(users):
                    user = int(user)
                    user_sim = user_similarity_dict.get(user, 0.5)
                    query_keys = self.KeyVAE.generate(last_keys[i].unsqueeze(0))
                    query_keys = torch.cat([last_keys[i].unsqueeze(0), query_keys.squeeze(0)], dim=0)
                    long_term_interest = self.memory.retrieve_memory(user, query_keys)
                    beta = self.beta + (user_sim - 0.5) * 0.5
                    interest[i] = long_term_interest * beta + short_term_interest[i] * (1 - beta)

                outputs = interest
                outputs[exclude_masks] = -1e10
                acc_5 += top_k_acc(labels, outputs, 5)
                acc_10 += top_k_acc(labels, outputs, 10)
                acc_20 += top_k_acc(labels, outputs, 20)
                mrr += mean_reciprocal_rank(labels, outputs)
                cnt += len(users)

            return acc_5 / cnt, acc_10 / cnt, acc_20 / cnt, mrr / cnt
