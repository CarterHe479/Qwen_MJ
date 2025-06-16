from rag_llm import ask_rag

while True:
    query = input("请输入问题：(输入exit退出)")
    if query.lower() == "exit":
        break

    result = ask_rag(query)
    raw = result["choices"][0]["text"].strip()

    # 保留“回答：”后的第一段，过滤多余问题
    if "回答：" in raw:
        raw = raw.split("回答：", 1)[-1].strip()
    clean = raw.split("问题：")[0].strip()  # 截断再次生成的问题

    print(f"\n🤖 回答：{clean}\n")
