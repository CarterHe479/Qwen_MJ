🧠 RAG-Qwen3: Retrieval-Augmented Generation with Qwen 0.6B
一个基于本地轻量级大模型 Qwen3-0.6B 的 RAG（检索增强生成）系统，支持 本地问答推理 + PDF 文档知识导入 + 向量检索 + rerank 精排 + prompt 压缩 + 可信度判断（拒答）。

本项目在 MacBook M2 Pro + 16GB 内存 下运行良好，适合本地学习/实验 RAG 技术。

🔧 技术架构
text
复制
编辑
          ┌───────────────┐
          │ 用户输入 Query│
          └──────┬────────┘
                 ↓
        ┌────────────────────┐
        │Query Embedding (text2vec) │
        └────────┬───────────┘
                 ↓
         ┌────────────────┐
         │ FAISS 相似度检索 │←──── 文档向量（由 SentenceTransformer 编码）
         └────────┬───────┘
                 ↓
        ┌──────────────────────┐
        │ 可选 rerank 精排 (bge-reranker) │
        └────────┬─────────────┘
                 ↓
        ┌─────────────────────┐
        │Prompt 构造（支持压缩+摘要）│
        └────────┬────────────┘
                 ↓
      ┌──────────────────────────────┐
      │Qwen3-0.6B 本地推理（llama-cpp-python）│
      └──────────────────────────────┘
📁 项目结构
bash
复制
编辑
llm_models/
├── qwen-0.6b-chat/                   # 本地 GGUF 模型存放目录
│   ├── Qwen3-0.6B-Q2_K.gguf
│   ├── Qwen3-0.6B-Q8_0.gguf
├── rag-qwen3/
│   ├── rag_demo.py                  # 主入口：交互式问答
│   ├── rag_llm.py                   # 检索 + Prompt 构造 + 信任机制
│   ├── rag_vectorstore.py          # 文档向量构建与索引
│   ├── pdf_to_chunks.py            # PDF 文档读取 + 分段处理
│   ├── rerank_utils.py             # reranker 精排模块（可选）
│   ├── summarizer.py               # Prompt压缩模块（可选）
│   ├── query_expander.py           # 查询扩展模块（可选）
│   ├── data/kb_docs.txt            # 切分后的知识文档（可替换）
│   ├── faiss.index + docs.pkl      # FAISS 索引 + 原文保存
🚀 功能特性
✅ 本地部署的小模型（Qwen3-0.6B）
无需联网，支持 CPU/M1/M2 本地运行。

使用 llama-cpp-python 加载 GGUF 格式。

✅ PDF 文档导入 + 标题感知切分
使用 PyMuPDF 提取文本

使用 langchain.text_splitter.RecursiveCharacterTextSplitter 按标题段落智能切分，并加重叠滑窗提升上下文连续性

✅ 检索增强（RAG）
使用 SentenceTransformer 生成文档向量

FAISS 向量数据库本地检索

支持 Query Expansion & 多向量融合（可选）

✅ rerank 精排器（可选）
使用 BAAI/bge-reranker-base 执行交互式 rerank 精排

得分用于重排序和过滤低置信度内容

✅ prompt 压缩（可选）
支持摘要化多个 chunk，压缩为单段 prompt 输入，节省 token，提高上下文聚焦能力

✅ 信任机制（低相关度拒答）
如果检索内容与 query 相似度过低（可设阈值），将拒绝构造 RAG prompt，输出“未使用知识库”的 LLM 回答

🛠️ 安装环境
使用 Conda 创建环境：

bash
复制
编辑
conda create -n qwen3-env python=3.10
conda activate qwen3-env
pip install -r requirements.txt
如果网络不畅，可使用清华/中科大镜像或手动下载模型到：

bash
复制
编辑
rag-qwen3/hf_model/bge-reranker-base/
🧪 运行 Demo
bash
复制
编辑
cd rag-qwen3
python3 rag_demo.py
输入自然语言问题，即可返回回答。

🧹 .gitignore 建议
gitignore
复制
编辑
# 缓存和临时文件
__pycache__/
*.py[cod]
*.pkl
*.index
*.DS_Store

# 模型文件
*.gguf
*.bin
*.pt

# PDF、向量库和文档数据
*.pdf
data/kb_docs.txt
docs.pkl
faiss.index

# HF 下载模型
rag-qwen3/hf_model/
📌 TODO
 支持基础检索与问答

 支持 PDF 文档输入

 Prompt 压缩

 reranker 精排

 查询扩展 + 向量融合（多 query 模式）

 信任机制 + 拒答

 WebUI 接口（如 gradio / streamlit）

 LangChain / RAGasLib 架构替代（可选）

📚 参考
Qwen3 by Alibaba: https://huggingface.co/Qwen

SentenceTransformers: https://www.sbert.net

FAISS: Facebook AI Similarity Search

bge-reranker: BAAI 开源精排器

llama-cpp-python: C++ 推理引擎绑定
