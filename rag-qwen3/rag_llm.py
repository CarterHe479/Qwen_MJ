import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

index = faiss.read_index("faiss.index")
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)

encoder = SentenceTransformer("shibing624/text2vec-base-multilingual")
llm = Llama(model_path="/Users/carterhe/Desktop/llm_models/qwen-0.6b-chat/Qwen3-0.6B-Q8_0.gguf", n_ctx=2048)

def search(query, top_k=3):
    query_embedding = encoder.encode([query], normalize_embedding=True)
    D, I = index.search(query_embedding, top_k)
    return [docs[i] for i in I[0]]

def ask_rag(query):
    retrieved_docs = search(query)
    context = "\n".join(retrieved_docs)
    prompt = f"请根据以下内容回答问题：\n\n{context}\n\n问题：{query}\n回答："
    result = llm(prompt, max_tokens=512, stop=["问题：", "Question:", "Q:", "选项：", "答案："])


    # return result["choices"][0]["text"].strip()
    return result
