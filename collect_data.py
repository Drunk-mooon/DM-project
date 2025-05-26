import os
import pandas as pd
import json
from collections import defaultdict
import re

def contains_garbage(text):
    text = text.lower()
    garbage_patterns = ["http", "www", "jpg", "jpeg", "png", "gif", "tiff", "bmp"]
    return any(pattern in text for pattern in garbage_patterns)

# Parameters
input_directory     = "./"    # CSV 文件目录
min_posts           = 20       # 用户最少发言数
min_subreddits      = 2       # 用户至少出现的不同 subreddit 数
max_users           = 5    # 最多选取的用户数
max_posts_per_user  = 50       # 每个用户最多保留的发言数

# CSV 列名
columns = [
    "text","id","subreddit","meta","time",
    "author","ups","downs","authorlinkkarma","authorkarma","authorisgold"
]

# 1. 聚合原始发言
user_posts = defaultdict(list)
for fn in os.listdir(input_directory):
    print(f"Processing {fn} ...")
    if not fn.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_directory, fn), header=2, names=columns)
    df = df.dropna()                                                  # 丢弃任意NaN
    df = df[df["text"].str.strip().astype(bool)]                       # 丢弃空 text
    df = df[~df["text"].apply(contains_garbage)]                       # 丢弃“乱码”
    for _, row in df.iterrows():
        user_posts[row["author"]].append(row.to_dict())

# 1.5 去重
deduped = {}
for user, posts in user_posts.items():
    seen, unique = set(), []
    for p in posts:
        t = p["text"].strip()
        if t not in seen:
            seen.add(t)
            unique.append(p)
    deduped[user] = unique

# 2. 双重筛选：发言数 & 不同 subreddit 数
qualified = {}
for user, posts in deduped.items():
    if len(posts) < min_posts:
        continue
    subs = {p["subreddit"] for p in posts}
    if len(subs) < min_subreddits:
        continue
    qualified[user] = posts

# 2.5 收集所有 meta 和 subreddit 值
all_meta      = sorted({p["meta"]      for posts in qualified.values() for p in posts})
all_subreddits= sorted({p["subreddit"] for posts in qualified.values() for p in posts})
meta2idx      = {m:i for i,m in enumerate(all_meta)}
sr2idx        = {s:i for i,s in enumerate(all_subreddits)}

# 3. 选取前 max_users 个用户
if(len(qualified) < max_users):
    print(f"Warning: Only {len(qualified)} users qualified, which is less than max_users={max_users}.")
    max_users = len(qualified)
selected = dict(list(qualified.items())[:max_users])

# 4. 构建输出
output = []
for uid, (user, posts) in enumerate(selected.items()):
    # 用户涉及的 meta/subreddit 集合
    user_metas = {p["meta"]      for p in posts}
    user_srs   = {p["subreddit"] for p in posts}

    # one-hot 向量
    meta_vector      = [1 if m in user_metas      else 0 for m in all_meta]
    subreddit_vector = [1 if s in user_srs       else 0 for s in all_subreddits]

    entry = {
        "user_id": uid,
        "user_name": user,
        "meta_vector": meta_vector,
        "subreddit_vector": subreddit_vector,
        "posts": posts[:max_posts_per_user]
    }
    output.append(entry)

# 保存
with open("./reddit_user_posts_filtered.json","w") as f:
    json.dump(output, f, indent=2)

print(f"Selected {len(output)} users, written to reddit_user_posts_filtered.json")
