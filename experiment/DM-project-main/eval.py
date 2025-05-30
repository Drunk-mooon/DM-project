import json
import argparse


def compute_global_hit_ratio(sub_file, rec_file, topk):
    # 加载 sub.json 并建立用户对集合（无序）
    with open(sub_file, "r", encoding="utf-8") as f:
        sub_data = json.load(f)

    user_pairs = sub_data.get("user_pairs", [])
    pair_set = set()
    for pair in user_pairs:
        u1 = pair["user1_id"]
        u2 = pair["user2_id"]
        pair_set.add((u1, u2))
        pair_set.add((u2, u1))

    # 加载 recommend.json
    with open(rec_file, "r", encoding="utf-8") as f:
        rec_data = json.load(f)

    total = 0
    hits = 0

    for user_str, recs in rec_data.items():
        user_id = int(user_str)
        for rec in recs[:topk]:
            node_id = rec["node"]
            total += 1
            if (user_id, node_id) in pair_set:
                hits += 1

    # 防止除以零
    hit_ratio = hits / total if total > 0 else 0
    return hits, total, hit_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Compute overall recommendation hit ratio."
    )
    parser.add_argument("--sub", type=str, required=True, help="Path to sub.json")
    parser.add_argument("--rec", type=str, required=True, help="Path to recommend.json")
    parser.add_argument(
        "--topk", type=int, required=True, help="Top K recommendations to consider"
    )
    args = parser.parse_args()

    hits, total, ratio = compute_global_hit_ratio(args.sub, args.rec, args.topk)
    print(f"Total recommendations checked: {total}")
    print(f"Matched user_pairs: {hits}")
    print(f"Overall hit ratio: {ratio:.4f}")


if __name__ == "__main__":
    main()
