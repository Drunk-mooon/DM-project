"""
Module: subreddit_overlap

This module writes out a JSON file containing two top-level keys:

1. subreddit_members
   - Type: dict[int, list[int]]
   - Maps each subreddit_id (the index in subreddit_vector) to a list of user_ids that belong to that subreddit.

2. user_pairs
   - Type: list[dict]
   - Each entry represents a pair of users with at least one subreddit in common.
   - Fields in each dict:
     • user1_id (int): the smaller user_id in the pair
     • user2_id (int): the larger user_id in the pair
     • overlap_count (int): number of shared subreddits
     • user1_sub_count (int): total subreddits for user1_id
     • user2_sub_count (int): total subreddits for user2_id
     • iou (float): intersection-over-union = overlap_count / (user1_sub_count + user2_sub_count - overlap_count)

Example output JSON structure:

{
  "subreddit_members": {
    "0": [123, 456, 789],
    "1": [456, 999],
    ...
  },
  "user_pairs": [
    {
      "user1_id": 123,
      "user2_id": 456,
      "overlap_count": 2,
      "user1_sub_count": 5,
      "user2_sub_count": 4,
      "iou": 2/ (5 + 4 - 2)
    },
    ...
  ]
}
"""

import argparse
import json
import os
from itertools import combinations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute subreddit membership and user-pair overlaps with IOU from Reddit user JSON data using subreddit_vector"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input JSON file containing list of user records",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    infile = args.input
    base, ext = os.path.splitext(infile)
    outfile = f"{base}_sub{ext}"

    # Load input data
    with open(infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build subreddit_id -> members mapping
    subreddit_members = {}
    # And user_id -> set of subreddit_ids
    user_subs = {}

    for record in data:
        # Use user_id as identifier
        user_id = record.get("user_id")
        if user_id is None:
            continue
        vec = record.get("subreddit_vector", [])
        # Indices with non-zero indicate membership
        subs = {i for i, v in enumerate(vec) if v}
        if not subs:
            continue
        user_subs[user_id] = subs
        for sub_id in subs:
            subreddit_members.setdefault(sub_id, []).append(user_id)

    # Precompute user subreddit counts
    user_counts = {uid: len(subs) for uid, subs in user_subs.items()}

    # Compute pairwise overlaps
    pair_overlap = {}
    for sub_id, members in subreddit_members.items():
        if len(members) < 2:
            continue
        for uid1, uid2 in combinations(sorted(members), 2):
            pair_overlap.setdefault((uid1, uid2), 0)
            pair_overlap[(uid1, uid2)] += 1

    # Assemble results with IOU
    user_pairs = []
    for (uid1, uid2), overlap in pair_overlap.items():
        count1 = user_counts.get(uid1, 0)
        count2 = user_counts.get(uid2, 0)
        union = count1 + count2 - overlap
        iou = overlap / union if union > 0 else 0
        user_pairs.append(
            {
                "user1_id": uid1,
                "user2_id": uid2,
                "overlap_count": overlap,
                "user1_sub_count": count1,
                "user2_sub_count": count2,
                "iou": iou,
            }
        )

    # Output structure
    output = {"subreddit_members": subreddit_members, "user_pairs": user_pairs}

    # Write to file
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote results to {outfile}")


if __name__ == "__main__":
    main()
