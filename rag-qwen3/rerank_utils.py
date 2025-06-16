from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载中文精排模型
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
model.eval()

def rerank(query: str, passages: list[str], top_k: int = 3) -> list[str]:
    pairs = [[query, p] for p in passages]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)  # shape: [N]
    sorted_indices = torch.argsort(scores, descending=True)
    return [passages[i] for i in sorted_indices[:top_k]]
