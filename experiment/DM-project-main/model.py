# 文件: model.py
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims=None):
        super(VAE, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        # 编码器
        modules = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Linear(last_dim, h_dim), nn.ReLU()))
            last_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        # 均值和对数方差层
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)

        # 解码器
        hidden_dims.reverse()
        modules = []
        last_dim = latent_dim
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Linear(last_dim, h_dim), nn.ReLU()))
            last_dim = h_dim
        self.decoder = nn.Sequential(*modules)
        # 重构层
        self.final_layer = nn.Linear(last_dim, input_dim)

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        x = self.decoder(z)
        return self.final_layer(x)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
