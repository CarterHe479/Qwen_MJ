from rag_llm import ask_rag

while True:
    query = input("请输入问题：(输入exit退出)")
    if query == "exit":
        break
    answer = ask_rag(query)
    text = answer["choices"][0]["text"].strip()
    print(text)
