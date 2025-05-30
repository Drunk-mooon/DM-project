#!/usr/bin/env python3
"""
Deep Embedded Clustering (DEC) Implementation with CLI

基于 Xie et al.，《Unsupervised Deep Embedding for Clustering Analysis》

CLI 参数:
  --emb EMB_PATH       输入 embedding .npy 文件路径
  -o, --output OUT_DIR 输出目录，保存模型与结果
  --clusters K         聚类簇数 (默认 10)
  --pretrain_epochs    自编码器预训练轮数 (默认 50)
  --fit_iters          DEC 迭代优化总次数 (默认 10000)
  --update_interval    更新目标分布间隔 (默认 140)
  --tol                收敛阈值 (默认 1e-3)

输出格式 (保存至 OUT_DIR):
  - latent.npy   用户潜在表示 Z (n_users, latent_dim)
  - q.npy        软分配概率 Q (n_users, n_clusters)
  - y_pred.npy   硬聚类标签 (n_users,)

示例:
  python dec.py --emb embeddings.npy -o results/ --clusters 5
"""

import os
import argparse
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import json


# ------------------------- 工具函数 -------------------------
def target_distribution(q: np.ndarray) -> np.ndarray:
    weight = q**2 / q.sum(axis=0)
    return (weight.T / weight.sum(axis=1)).T


# ------------------------- 模型定义 -------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()
        encoder, decoder = [], []
        last = input_dim
        for h in hidden_dims:
            encoder += [nn.Linear(last, h), nn.ReLU()]
            last = h
        decoder_dims = hidden_dims[::-1] + [input_dim]
        for h in decoder_dims[:-1]:
            decoder += [nn.Linear(last, h), nn.ReLU()]
            last = h
        decoder += [nn.Linear(last, decoder_dims[-1])]
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class DEC:
    def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters, device):
        self.device = torch.device(device)
        # 构造自编码器：hidden_dims + [latent_dim]
        self.ae = Autoencoder(input_dim, hidden_dims + [latent_dim]).to(self.device)
        self.encoder = self.ae.encoder
        self.n_clusters = n_clusters
        self.cluster_centers = None

    def pretrain(self, x, epochs, batch_size, lr):
        x_t = torch.tensor(x, dtype=torch.float32).to(self.device)
        loader = DataLoader(TensorDataset(x_t), batch_size=batch_size, shuffle=True)
        opt = optim.Adam(self.ae.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.ae.train()
        for ep in range(epochs):
            tot = 0
            for (batch,) in loader:
                recon, _ = self.ae(batch)
                loss = criterion(recon, batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * batch.size(0)
            print(f"Pretrain {ep + 1}/{epochs}, Loss={tot / len(x):.4f}")

    def initialize_cluster(self, x):
        self.ae.eval()
        with torch.no_grad():
            z = (
                self.encoder(torch.tensor(x, dtype=torch.float32).to(self.device))
                .cpu()
                .numpy()
            )
        km = KMeans(n_clusters=self.n_clusters, n_init=20)
        y = km.fit_predict(z)
        self.cluster_centers = torch.tensor(
            km.cluster_centers_, dtype=torch.float32
        ).to(self.device)
        return y

    def fit(self, x, iters, update_interval, tol, batch_size, lr):
        """
        输出含义见 DEC.md
        """
        X = torch.tensor(x, dtype=torch.float32).to(self.device)
        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
        opt = optim.Adam(self.encoder.parameters(), lr=lr)
        y_last = None

        for it in range(iters):
            if it % update_interval == 0:
                with torch.no_grad():
                    z = self.encoder(X).cpu().numpy()
                dist = np.sum(
                    (z[:, None] - self.cluster_centers.cpu().numpy()[None]) ** 2, axis=2
                )
                q = 1.0 / (1.0 + dist)
                q = (q ** ((1 + 1) / 2)) / np.sum(
                    q ** ((1 + 1) / 2), axis=1, keepdims=True
                )
                p = target_distribution(q)
                y = q.argmax(axis=1)
                if y_last is not None and np.sum(y != y_last) / len(y) < tol:
                    print(f"Converged @ iter {it}")
                    break
                y_last = y
                self.encoder.train()

            for batch_idx, (batch,) in enumerate(loader):
                z_b = self.encoder(batch)
                # 取对应 p_batch
                start = batch_idx * batch.size(0)
                end = start + batch.size(0)
                p_b = torch.tensor(p[start:end], dtype=torch.float32).to(self.device)
                dist_b = torch.sum(
                    (z_b[:, None] - self.cluster_centers[None]) ** 2, dim=2
                )
                q_b = (1.0 + dist_b).pow(-1)
                q_b = q_b / torch.sum(q_b, dim=1, keepdim=True)
                loss = nn.KLDivLoss(reduction="batchmean")(q_b.log(), p_b)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if it % update_interval == 0:
                print(f"Iter {it}, KL Loss={loss.item():.4f}")
        # 最终 soft assignments & latent
        self.ae.eval()
        with torch.no_grad():
            z_final = self.encoder(X).cpu().numpy()
        dist = np.sum(
            (z_final[:, None] - self.cluster_centers.cpu().numpy()[None]) ** 2, axis=2
        )
        q_final = 1.0 / (1.0 + dist)
        q_final = (q_final ** ((1 + 1) / 2)) / np.sum(
            q_final ** ((1 + 1) / 2), axis=1, keepdims=True
        )
        y_pred = q_final.argmax(axis=1)
        return z_final, q_final, y_pred


# ------------------------- CLI -------------------------
def main():
    parser = argparse.ArgumentParser(description="DEC Clustering CLI")
    parser.add_argument("--emb", required=True, help="输入 embedding .npy 路径")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--clusters", type=int, default=10, help="簇数")
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--fit_iters", type=int, default=3000)
    parser.add_argument("--update_interval", type=int, default=140)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # 载入数据
    x = np.load(args.emb)
    os.makedirs(args.output, exist_ok=True)

    # 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dec = DEC(
        input_dim=x.shape[1],
        hidden_dims=[500, 500, 2000],
        latent_dim=10,
        n_clusters=args.clusters,
        device=device,
    )

    # 预训练 & 初始化 & 训练
    dec.pretrain(x, args.pretrain_epochs, args.batch_size, args.lr)
    dec.initialize_cluster(x)
    z, q, y = dec.fit(
        x, args.fit_iters, args.update_interval, args.tol, args.batch_size, args.lr
    )

    # 保存结果
    np.save(os.path.join(args.output, "latent.npy"), z)
    np.save(os.path.join(args.output, "q.npy"), q)
    np.save(os.path.join(args.output, "y_pred.npy"), y)

    # 保存 y_pred 为 JSON 格式：node_id → cluster_id
    y_dict = {str(i): int(label) for i, label in enumerate(y)}
    with open(os.path.join(args.output, "y_pred.json"), "w") as f:
        json.dump(y_dict, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
