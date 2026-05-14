from sentence_transformers import SentenceTransformer
import numpy as np

print("加载模型...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentences = [
    "How does chaining resolve collisions?",
    "除法雜湊法怎麼運作?"
]

print("计算 embedding...")
embeddings = model.encode(sentences)

print(f"\n✓ 模型加载成功!")
print(f"  向量维度: {embeddings.shape}")
print(f"  第一个向量的范数: {np.linalg.norm(embeddings[0]):.4f}")
print(f"  第一个向量前5个值: {embeddings[0][:5]}")

# 计算两个句子的相似度（应该 > 0.5 因为都关于哈希）
similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
print(f"  两个句子的余弦相似度: {similarity:.4f}")