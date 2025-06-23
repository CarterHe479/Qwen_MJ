import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from rerank_utils import rerank
from query_expander import expand_query
from summarizer import summarize
from sentence_transformers import util

index = faiss.read_index("faiss.index")
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)

encoder = SentenceTransformer("shibing624/text2vec-base-multilingual")
llm = Llama(model_path="/Users/carterhe/Desktop/llm_models/qwen-0.6b-chat/Qwen3-0.6B-Q8_0.gguf", n_ctx=2048)

RAG_THRESHOLD = 0.3

def search(query, top_k=3):
    query_embedding = encoder.encode([query], normalize_embedding=True)
    D, I = index.search(query_embedding, top_k)
    return [docs[i] for i in I[0]]

def search_fused(query, top_k=3):
    expanded_queries = expand_query(query)

    all_chunks = []
    for q in expanded_queries:
        emb = encoder.encode([q], normalize_embeddings=True)
        D, I = index.search(emb, top_k)
        for idx in I[0]:
            all_chunks.append(docs[idx])

    # 去重
    all_chunks = list(dict.fromkeys(all_chunks))  # 保留顺序
    return all_chunks

def check_similarity_threshold(query, docs, encoder, threshold=0.3):
    "返回query和top 1 chunk的相似度是否达标"
    query_emb = encoder.encode([query], normalize_embeddings=True)
    docs_embs = encoder.encode(docs, normalize_embeddings=True)

    sims = util.cos_sim(query_emb, docs_embs)[0]
    max_sim = sims.max().item()

    return max_sim


def ask_rag(query):
    retrieved_docs = search_fused(query)
    sim_scores = check_similarity_threshold(query, retrieved_docs, encoder)

    if sim_scores >= RAG_THRESHOLD:
        # 使用引入的重排模型
        reranked_docs = rerank(query, retrieved_docs, top_k=3)
        long_context = "\n".join(reranked_docs)
        context = summarize(long_context)
        prompt = f"请根据以下内容回答问题：\n\n{context}\n\n问题：{query}\n回答："
        result = llm(prompt, max_tokens=512, stop=["问题：", "Question:", "Q:", "选项：", "答案："])


        # return result["choices"][0]["text"].strip()
        return result
    else:
        print("⚠️ 相似度较低，以下回答未使用RAG。")
        prompt = f"{query}\n回答："
        result = llm(prompt, max_tokens=512, stop=["问题：", "选项：", "答案："])
        return {
            "choices": [{
                "text": "[未使用RAG] " + result["choices"][0]["text"].strip()
            }]
        }
