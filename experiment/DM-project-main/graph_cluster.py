#!/usr/bin/env python3
"""
从 embedding 构建 kNN 图并执行 Louvain、Label Propagation 与 Infomap 社区发现

输入:
  --emb    .npy 文件路径，格式为 (n_users, emb_dim)
  -o/--output 输出目录
  --k      构图近邻数

输出:
  louvain.json        节点 → 社区映射（Louvain）
  labelprop.json      节点 → 社区映射（Label Propagation）
  infomap.json        节点 → 社区映射（Infomap）

依赖:
  - networkx
  - sklearn
  - python-louvain (pip install python-louvain)
  - infomap (pip install infomap)
"""

import argparse
import numpy as np
import networkx as nx
import json
import os

from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain  # pip install python-louvain
from networkx.algorithms.community import label_propagation_communities
from collections import defaultdict

# Infomap 引入
try:
    from infomap import Infomap
except ImportError:
    Infomap = None


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


def build_knn_graph(emb: np.ndarray, k: int) -> nx.Graph:
    n_users = emb.shape[0]
    sim_matrix = cosine_similarity(emb)
    np.fill_diagonal(sim_matrix, 0)
    G = nx.Graph()
    G.add_nodes_from(range(n_users))
    for i in range(n_users):
        neighbors = np.argpartition(-sim_matrix[i], k)[:k]
        for j in neighbors:
            weight = float(sim_matrix[i, j])
            if G.has_edge(i, j):
                if G[i][j]["weight"] < weight:
                    G[i][j]["weight"] = weight
            else:
                G.add_edge(i, j, weight=weight)
    return G


def run_louvain(G: nx.Graph) -> dict:
    partition = community_louvain.best_partition(G, weight="weight")
    return {str(k): int(v) for k, v in partition.items()}


def run_label_propagation(G: nx.Graph) -> dict:
    communities = label_propagation_communities(G)
    label_map = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            label_map[str(node)] = cid
    return label_map


def run_infomap(G: nx.Graph) -> dict:
    if Infomap is None:
        raise ImportError(
            "Infomap package not installed. Please install with `pip install infomap`."
        )
    im = Infomap()
    # 将图边添加到 Infomap
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1.0)
        im.addLink(int(u), int(v), weight)
    im.run()
    label_map = {}
    for node in im.iterTree():
        if node.isLeaf:
            label_map[str(node.physicalId)] = node.moduleIndex()
    return label_map


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def community_to_cluster_format(
    partition: dict[str, int],
) -> dict[str, list[dict[str, float]]]:
    """
    将节点 → 社区 映射 转为 cluster 格式输出：
    同一社区的节点互为相似邻居，sim = 1.0
    """
    community_map = defaultdict(list)
    for node, cid in partition.items():
        community_map[cid].append(int(node))

    cluster_result = defaultdict(list)
    for members in community_map.values():
        for i in members:
            cluster_result[str(i)].extend(
                {"node": int(j), "sim": 1.0} for j in members if j != i
            )
    return cluster_result


def main():
    parser = argparse.ArgumentParser(
        description="社区发现 (Louvain / Label Propagation / Infomap)"
    )
    parser.add_argument("--emb", required=True, help="输入 embedding 的 .npy 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--k", type=int, default=10, help="构图近邻数")
    parser.add_argument(
        "--cluster", action="store_true", help="是否输出 _cluster.json 格式的相似节点"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    emb = load_embeddings(args.emb)
    G = build_knn_graph(emb, args.k)

    print("运行 Louvain 社区发现...")
    louvain_result = run_louvain(G)
    save_json(louvain_result, os.path.join(args.output, "louvain.json"))
    if args.cluster:
        louvain_cluster = community_to_cluster_format(louvain_result)
        save_json(louvain_cluster, os.path.join(args.output, "louvain_cluster.json"))

    print("运行 Label Propagation 社区发现...")
    labelprop_result = run_label_propagation(G)
    save_json(labelprop_result, os.path.join(args.output, "labelprop.json"))
    if args.cluster:
        labelprop_cluster = community_to_cluster_format(labelprop_result)
        save_json(
            labelprop_cluster, os.path.join(args.output, "labelprop_cluster.json")
        )

    print("运行 Infomap 社区发现...")
    try:
        infomap_result = run_infomap(G)
        save_json(infomap_result, os.path.join(args.output, "infomap.json"))
        if args.cluster:
            infomap_cluster = community_to_cluster_format(infomap_result)
            save_json(
                infomap_cluster, os.path.join(args.output, "infomap_cluster.json")
            )
    except ImportError as e:
        print(e)

    print(f"输出已保存至 {args.output}")


if __name__ == "__main__":
    main()
