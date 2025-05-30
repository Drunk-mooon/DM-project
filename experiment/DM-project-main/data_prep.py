import json
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Extract user_id and personality_vector from JSON file."
    )
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument(
        "--fill_notfound",
        required=True,
        type=float,
        help="Fill missing vectors with DEFAULT_VALUE",
    )

    args = parser.parse_args()
    input_path = args.input

    # 生成输出文件路径
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_vectors{ext}"

    # 读取 JSON 数据
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取 user_id 和 personality_vector，并排序
    # 记录缺少 personality_vector 的用户
    missing_users = [
        user["user_id"] for user in data if "personality_vector" not in user
    ]
    if missing_users:
        print(f"Users missing personality_vector: {missing_users}")
        if args.fill_notfound:
            DEFAULT_VALUE = args.fill_notfound
            print(f"Fill missing vectors with {DEFAULT_VALUE}")
        else:
            # hint user and quit
            print("Missing vectors, use --fill_notfound to fill with DEFAULT_VALUE")
            exit(1)

    vector_len = len(data[0]["personality_vector"])
    # 如果缺失vector，就填充为DEFAULT_VALUE为值的vector。长度和其他相同
    for user in data:
        if "personality_vector" not in user:
            # 长度和其他长度相同
            user["personality_vector"] = [DEFAULT_VALUE] * vector_len
    user_vectors = {user["user_id"]: user["personality_vector"] for user in data}

    sorted_user_vectors = dict(sorted(user_vectors.items()))

    # 写入输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_user_vectors, f, ensure_ascii=False, indent=2)

    # 同时输出为一个(n_users, n_embedding)的np.ndarray
    import numpy as np

    np_path = f"{base}_np.npy"
    np.save(np_path, np.array(list(sorted_user_vectors.values())).astype(np.float32))

    print(f"Vectors written to {output_path}")
    print(f"NumPy array written to {np_path}")


if __name__ == "__main__":
    main()
