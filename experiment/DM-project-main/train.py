# 文件: train.py
import argparse
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import VAE
from torch import nn


def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss(reduction="sum")(recon_x, x)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


def train(args):
    device = torch.device("cuda:0")
    # 加载数据
    data = torch.from_numpy(np.load(args.data_path)).float()
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    vae = VAE(input_dim=data.size(1), latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        vae.train()
        train_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(x)
            loss = loss_function(recon, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {train_loss / len(dataset):.4f}")
    # 保存模型
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(vae.state_dict(), os.path.join(args.out_dir, "vae_checkpoint.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="输入 embedding 的 .npy 文件路径"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./checkpoints", help="模型保存目录"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=16)
    args = parser.parse_args()
    import numpy as np

    train(args)
