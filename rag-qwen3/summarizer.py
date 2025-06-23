from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Randeng-T5-784M-Summary-Chinese")
model = AutoModelForSeq2SeqLM.from_pretrained("IDEA-CCNL/Randeng-T5-784M-Summary-Chinese")

def summarize(text: str, max_len=150):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True)
    output = model.generate(input_ids, max_new_tokens=max_len)
    return tokenizer.decode(output[0], skip_special_tokens=True)
