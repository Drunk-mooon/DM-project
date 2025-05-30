import os
import pandas as pd
import json
import random
from collections import defaultdict

def contains_garbage(text):
    text = text.lower()
    garbage_patterns = ["http", "www", "jpg", "jpeg", "png", "gif", "tiff", "bmp"]
    return any(pattern in text for pattern in garbage_patterns)

# Parameters
input_directory     = "./raw_data"    # CSV 文件目录
min_posts           = 1      # 用户最少发言数 k
min_subreddits      = 1      # 用户至少出现的不同 subreddit 数
max_users           = 50000       # 最多选取的用户数 m
max_posts_per_user  = 500      # 每个用户最多保留的发言数

# CSV 列名
columns = [
    "text","id","subreddit","meta","time",
    "author","ups","downs","authorlinkkarma","authorkarma","authorisgold"
]

# 1. 读入 & 清洗 & 聚合
user_posts = defaultdict(list)
for fn in os.listdir(input_directory):
    print(f"Processing {fn}...")
    if not fn.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(input_directory, fn), header=2, names=columns)
    df = df.dropna()                                    # 丢弃任意 NaN
    df = df[df["text"].str.strip().astype(bool)]         # 丢弃空白 text
    df = df[~df["text"].apply(contains_garbage)]         # 丢弃“乱码”
    for _, row in df.iterrows():
        user_posts[row["author"]].append(row.to_dict())

# 2. 去重（text 字段）
deduped = {}
for user, posts in user_posts.items():
    seen, unique = set(), []
    for p in posts:
        t = p["text"].strip()
        if t not in seen:
            seen.add(t)
            unique.append(p)
    deduped[user] = unique

# 3. 双筛选：发言数 & 不同 subreddit 数
qualified = {}
for user, posts in deduped.items():
    if len(posts) < min_posts:
        continue
    subs = {p["subreddit"] for p in posts}
    if len(subs) < min_subreddits:
        continue
    qualified[user] = posts

# 如果没有足够用户给出警告
if not qualified:
    raise RuntimeError("No users meet the criteria.")

# 4. Shuffle users
user_items = list(qualified.items())
random.shuffle(user_items)

# 5. 按 (subreddit 数, 发言数) 降序排序
def sort_key(item):
    user, posts = item
    num_subs  = len({p["subreddit"] for p in posts})
    num_posts = len(posts)
    return (num_posts, num_subs)

user_items.sort(key=sort_key, reverse=True)

# 6. 限制最多 max_users
selected_items = user_items[:max_users]

# 7. 收集所有 meta & subreddit 用于 vector
all_meta       = sorted({p["meta"]      for _, posts in selected_items for p in posts})
all_subreddits = sorted({p["subreddit"] for _, posts in selected_items for p in posts})

# 8. 构建最终输出
output = []
for uid, (user, posts) in enumerate(selected_items):
    user_metas = {p["meta"]      for p in posts}
    user_srs   = {p["subreddit"] for p in posts}

    meta_vector      = [1 if m in user_metas else 0 for m in all_meta]
    subreddit_vector = [1 if s in user_srs   else 0 for s in all_subreddits]

    entry = {
        "user_id": uid,
        "user_name": user,
        "meta_vector": meta_vector,
        "subreddit_vector": subreddit_vector,
        "posts": posts[:max_posts_per_user]
    }
    output.append(entry)

# 9. 写入 JSON
with open("reddit_user_posts_filtered.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Selected {len(output)} users, saved to reddit_user_posts_filtered.json")
