# 文件: transform_embedding.py
import torch
import numpy as np
from model import VAE


def transform(data_path: str, checkpoint_path: str, output_path: str):
    """
    更健壮的 checkpoint 加载：加载 VAE 模型时使用 strict=False 忽略不匹配的层，
    并提取 encoder 输出的 mu 作为新的嵌入表示。

    Args:
        data_path: .npy 格式的输入 embedding 路径
        checkpoint_path: 保存的 VAE 模型 checkpoint 路径
        output_path: 输出新的 latent 嵌入 .npy 路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    embeddings = torch.from_numpy(np.load(data_path)).float().to(device)

    # 读取 checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)

    # 推断模型维度：取输入数据维度和 checkpoint 中的 latent_dim
    input_dim = embeddings.size(1)
    # 从 fc_mu.weight 推断 latent_dim
    latent_dim = state_dict["fc_mu.weight"].size(0)

    # 构建模型并加载 state_dict（允许忽略尺寸不匹配的层）
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
    load_result = vae.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"警告：以下键未加载到模型中：{load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"警告：加载时遇到多余的键：{load_result.unexpected_keys}")
    vae.to(device).eval()

    # 提取新的隐空间 embedding
    with torch.no_grad():
        mu, _ = vae.encode(embeddings)
    latent = mu.cpu().numpy()

    # 保存新的 embedding
    np.save(output_path, latent)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    transform(args.data_path, args.checkpoint_path, args.output_path)
