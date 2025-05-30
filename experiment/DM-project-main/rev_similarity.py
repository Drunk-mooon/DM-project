#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import json
from typing import List, Tuple, Dict
from collections import defaultdict


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


def compute_similarity_pairs(
    embeddings: np.ndarray, metric: str
) -> List[Tuple[int, int, float]]:
    n_users = embeddings.shape[0]
    eps = 1e-8

    iu, ju = np.triu_indices(n_users, k=1)

    if metric == "cosine":
        norms = np.linalg.norm(embeddings, axis=1)
        dots = (embeddings @ embeddings.T)[iu, ju]
        denom = (norms[iu] * norms[ju]) + eps
        sims = dots / denom

    elif metric == "euclidean":
        diff = embeddings[iu] - embeddings[ju]
        dists = np.linalg.norm(diff, axis=1)
        sims = 1.0 / (1.0 + dists)  # 距离越小相似度越大

    elif metric == "jaccard":
        mean = embeddings.mean(axis=0)
        std = embeddings.std(axis=0)
        normed = (embeddings - mean) / (std + eps)
        bin_emb = (normed > 0).astype(int)

        inter = (bin_emb @ bin_emb.T)[iu, ju].astype(float)
        unions = bin_emb.sum(axis=1)[iu] + bin_emb.sum(axis=1)[ju] - inter + eps
        sims = inter / unions
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return list(zip(iu.tolist(), ju.tolist(), sims.tolist()))


def build_botk_dict(
    sim_pairs: List[Tuple[int, int, float]], n_users: int, botk: int
) -> Dict[int, List[Dict[str, float]]]:
    sim_dict = defaultdict(list)
    for i, j, sim in sim_pairs:
        sim_dict[i].append((j, sim))
        sim_dict[j].append((i, sim))

    botk_dict = {
        node: sorted(neighbors, key=lambda x: x[1])[:botk]
        for node, neighbors in sim_dict.items()
    }

    return {
        str(node): [{"node": int(nei), "sim": float(sim)} for nei, sim in neighbors]
        for node, neighbors in botk_dict.items()
    }


def main():
    parser = argparse.ArgumentParser(description="用户嵌入相似度最低 botk 输出")
    parser.add_argument(
        "--emb", type=str, required=True, help="输入 embedding .npy 文件路径"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="输出 JSON 文件路径，默认 stdout"
    )
    parser.add_argument(
        "--botk", type=int, default=10, help="每个节点保留的相似度最低的 k 个节点"
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=["cosine", "euclidean", "jaccard"],
        help="相似度度量方法",
    )
    args = parser.parse_args()

    emb = load_embeddings(args.emb)
    sim_pairs = compute_similarity_pairs(emb, args.metric)
    botk_result = build_botk_dict(sim_pairs, emb.shape[0], args.botk)

    output_str = json.dumps(botk_result, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
    else:
        sys.stdout.write(output_str)


if __name__ == "__main__":
    main()
