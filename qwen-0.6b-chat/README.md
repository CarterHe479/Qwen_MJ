# 🔍 RAG-Qwen3: 本地中文 PDF 知识问答系统

本项目实现了一个轻量级、本地运行的 RAG（Retrieval-Augmented Generation）系统，支持通过 PDF 文档构建知识库，并使用本地 Qwen3-0.6B 模型进行中文问答。

## ✅ 项目特点

- 🧠 支持中文语义检索（基于 GTE 向量模型）
- 📄 支持 PDF 文档导入自动构建知识库
- 💬 支持本地 LLM 回答（llama-cpp-python + GGUF 模型）
- ⚡ 适配 MacBook M 系列 / 低资源环境
- 🛠️ 基于 FAISS 向量检索构建 RAG 原型

## 📁 项目结构

```text
├── qwen-0.6b-chat
│   ├── qwen_test.py
│   ├── Qwen3-0.6B-Q2_K.gguf
│   └── Qwen3-0.6B-Q8_0.gguf
└── rag-qwen3
    ├── __pycache__
    │   ├── pdf_to_chunks.cpython-310.pyc
    │   └── rag_llm.cpython-310.pyc
    ├── data
    │   └── kb_docs.txt
    ├── docs.pkl
    ├── example.pdf
    ├── faiss.index
    ├── pdf_to_chunks.py
    ├── rag_demo.py
    ├── rag_llm.py
    ├── rag_vectorstore.py
    └── requirements.txt
