from sentence_transformers import SentenceTransformer
import faiss
import pickle

# 载入数据
with open("data/kb_docs.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f.readlines()]

# 使用轻量模型（bge-small、e5-small-v2 都可以）
encoder = SentenceTransformer("thenlper/gte-small")  # 支持中文
embeddings = encoder.encode(texts, normalize_embeddings=True)

# 建 FAISS 索引
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# 保存索引与原始数据
faiss.write_index(index, "faiss.index")
with open("docs.pkl", "wb") as f:
    pickle.dump(texts, f)
