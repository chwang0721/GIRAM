import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyVAE(nn.Module):
    def __init__(self, input_dim, n_samples, latent_dim=128, hidden_dim=128):
        super(KeyVAE, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + self.condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.n_samples = n_samples

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, mu, logvar, condition):
        esps = torch.randn(mu.size(0), self.n_samples, self.latent_dim, device=mu.device)
        std = torch.exp(0.5 * logvar)
        zs = mu.unsqueeze(1) + esps * std.unsqueeze(1)  # shape: (batch_size, self.n_samples, latent_dim)
        conditions = condition.unsqueeze(1).repeat(1, self.n_samples, 1)
        z_condition = torch.cat([zs, conditions], dim=-1)
        z_condition = z_condition.view(-1, self.latent_dim + self.condition_dim)
        generated_keys = self.decoder(z_condition)
        return generated_keys.view(-1, self.n_samples, self.input_dim)

    def forward(self, x):
        mu, logvar = self.encode(x)
        generated_key = self.decode(mu, logvar, x)
        return generated_key, mu, logvar

    def generate(self, condition):
        zs = torch.randn(condition.size(0), self.n_samples, self.latent_dim, device=condition.device)
        conditions = condition.unsqueeze(1).repeat(1, self.n_samples, 1)
        z_condition = torch.cat([zs, conditions], dim=-1)
        z_condition = z_condition.view(-1, self.latent_dim + self.condition_dim)
        generated_keys = self.decoder(z_condition)
        return generated_keys.view(-1, self.n_samples, self.input_dim)

    def KeyVAE_loss(self, generated_keys, original, mu, logvar):
        original = original.unsqueeze(1).repeat(1, generated_keys.size(1), 1)

        # Compute pairwise distances for all batches
        pairwise_distances = torch.cdist(generated_keys, generated_keys, p=2)
        diversity_matrix = 1.0 / (pairwise_distances + 1e-8).pow(2)
        diversity_matrix = diversity_matrix - torch.diag_embed(torch.diagonal(diversity_matrix, dim1=-2, dim2=-1))
        diversity_loss = diversity_matrix.sum(dim=(-2, -1)) / (self.n_samples * (self.n_samples - 1))  # Normalize per batch
        diversity_loss = diversity_loss.mean()

        recon_loss = F.mse_loss(generated_keys, original, reduction="mean")
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_div + diversity_loss * 0.1
        return loss
