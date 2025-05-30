import numpy as np

# 生成 (1000, 8) 随机 embedding
rng = np.random.RandomState(42)
embeddings = rng.randn(1000, 8)

# 保存为 .npy 文件
file_path = "./embeddings.npy"
np.save(file_path, embeddings)

# 显示形状和前 5 行
print("生成的 embedding 形状:", embeddings.shape)
print("前 5 行示例：")
print(embeddings[:5])

# 提示文件保存位置
print(f"\nembedding 已保存至：{file_path}")
