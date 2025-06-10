from llama_cpp import Llama

MODEL_PATH = "/Users/carterhe/Desktop/llm_models/qwen-0.5b-chat/Qwen3-0.6B-Q8_0.gguf"


# 初始化 LLM（n_ctx 表示上下文长度；n_threads 按你 CPU 核心设定）
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=6)

# 示例输入
output = llm("用一句话介绍你自己", max_tokens=100)
print(output["choices"][0]["text"].strip())

