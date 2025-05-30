import json
import argparse
import random


def compute_random_hit_ratio(sub_file, topk):
    # 加载 sub.json 并建立用户对集合（无序）
    with open(sub_file, "r", encoding="utf-8") as f:
        sub_data = json.load(f)

    user_pairs = sub_data.get("user_pairs", [])
    pair_set = set()
    user_set = set()

    for pair in user_pairs:
        u1 = pair["user1_id"]
        u2 = pair["user2_id"]
        pair_set.add((u1, u2))
        pair_set.add((u2, u1))
        user_set.update([u1, u2])

    user_list = list(user_set)
    total = 0
    hits = 0

    # 为每个用户随机推荐 topk 个用户（不能是自己）
    for user_id in user_list:
        candidates = [u for u in user_list if u != user_id]
        if len(candidates) < topk:
            top_recs = candidates  # 所有候选人都推荐
        else:
            top_recs = random.sample(candidates, topk)

        for rec_user_id in top_recs:
            total += 1
            if (user_id, rec_user_id) in pair_set:
                hits += 1

    hit_ratio = hits / total if total > 0 else 0
    return hits, total, hit_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Compute overall hit ratio using random recommendations."
    )
    parser.add_argument("--sub", type=str, required=True, help="Path to sub.json")
    parser.add_argument(
        "--topk",
        type=int,
        required=True,
        help="Top K random recommendations to consider",
    )
    args = parser.parse_args()

    hits, total, ratio = compute_random_hit_ratio(args.sub, args.topk)
    print(f"Total recommendations checked: {total}")
    print(f"Matched user_pairs: {hits}")
    print(f"Overall hit ratio: {ratio:.4f}")


if __name__ == "__main__":
    main()
