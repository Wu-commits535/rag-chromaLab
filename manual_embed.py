from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def encode(sentences):
    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        outputs = model(**encoded)
    # mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    # normalize
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    return embeddings.numpy()

sentences = [
    "How does chaining resolve collisions?",
    "除法雜湊法怎麼運作?"
]

embeddings = encode(sentences)
print(f"向量维度: {embeddings.shape}")
print(f"第一个向量前5个值: {embeddings[0][:5]}")

similarity = np.dot(embeddings[0], embeddings[1])
print(f"余弦相似度: {similarity:.4f}")