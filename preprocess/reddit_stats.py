#!/usr/bin/env python3
import os
import glob
import argparse
from collections import Counter

import pandas as pd

def gather_author_counts(data_dir):
    """
    遍历 data_dir 下所有 .csv 文件，读取 author 列，累加到 Counter 中。
    """
    counter = Counter()
    pattern = os.path.join(data_dir, '*.csv')
    for filepath in glob.glob(pattern):
        try:
            # 只读取 author 列，加快速度，省内存
            df = pd.read_csv(filepath, usecols=['5'])
        except Exception as e:
            print(f"跳过文件 {filepath}，读取出错：{e}")
            continue
        # 丢弃缺失值并累加计数
        counter.update(df['5'].dropna().tolist())
    return counter

def compute_stats(counter, k):
    total_users = len(counter)
    total_comments = sum(counter.values())

    # 1. 是否存在某用户多条发言
    has_multiple = any(count > 1 for count in counter.values())

    # 2. 每个用户平均发表多少条
    avg_comments_per_user = total_comments / total_users if total_users else 0

    # 3. 最多发表言论的用户发了多少条
    max_comments = max(counter.values()) if counter else 0

    # 4. 发言数 >= k 的用户数
    users_ge_k = sum(1 for count in counter.values() if count >= k)

    return {
        'has_multiple': has_multiple,
        'avg_comments_per_user': avg_comments_per_user,
        'max_comments': max_comments,
        'users_ge_k': users_ge_k,
        'total_users': total_users,
        'total_comments': total_comments,
    }

def main():
    parser = argparse.ArgumentParser(
        description="统计 Reddit Comment Dataset 中用户发言分布情况"
    )
    parser.add_argument(
        '--data_dir', '-d',
        required=True,
        help="存放 `<metareddit>_<subreddit>.csv` 文件的目录路径"
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help="统计发言数 ≥ k 的用户数量，默认 k=5"
    )
    args = parser.parse_args()

    print(f"正在扫描目录：{args.data_dir}，阈值 k={args.k}\n")
    counter = gather_author_counts(args.data_dir)
    stats = compute_stats(counter, args.k)

    print("=== 统计结果 ===")
    print(f"总用户数：{stats['total_users']}")
    print(f"总评论数：{stats['total_comments']}")
    print(f"是否存在发言超过 1 条的用户？{'是' if stats['has_multiple'] else '否'}")
    print(f"每个用户平均发表评论数：{stats['avg_comments_per_user']:.2f}")
    print(f"最多发言的用户发表了 {stats['max_comments']} 条评论")
    print(f"发言数 ≥ {args.k} 的用户共有：{stats['users_ge_k']} 个")

if __name__ == '__main__':
    main()
