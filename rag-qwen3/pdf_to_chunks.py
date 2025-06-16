import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            texts.append(text.strip())
    return "\n".join(texts)

# def split_text(text, max_chunk_len=300):
#     chunks = []
#     paragraph = ""
#     for line in text.split("\n"):
#         if len(paragraph) + len(line) < max_chunk_len:
#             paragraph += line.strip() + " "
#         else:
#             chunks.append(paragraph.strip())
#             paragraph = line.strip() + " "
#     if paragraph:
#         chunks.append(paragraph.strip())
#     return chunks

def smart_split(text, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "。", "！", "？", "；"]
    )
    paragraphs = custom_title_aware_split(text)

    chunks = []
    for para in paragraphs:
        chunks.extend(splitter.split_text(para))  # 每段分别切分
    return chunks



def custom_title_aware_split(text):
    lines = text.split("\n")
    paragraphs = []
    buffer = ""

    for line in lines:
        # 识别是否是标题：如全大写或带数字编号开头
        is_title = bool(re.match(r"^\s*(第?\d+章|[A-Z\s]+|[一二三四五六七八九十]+、)", line.strip()))
        if is_title and buffer:
            paragraphs.append(buffer.strip())
            buffer = line.strip()
        else:
            buffer += "\n" + line.strip()
    if buffer:
        paragraphs.append(buffer.strip())
    return paragraphs


def convert_pdf_to_chunks(pdf_path, output_txt="data/kb_docs.txt"):
    text = extract_text_from_pdf(pdf_path)
    # chunks = split_text(text)
    chunks = smart_split(text)
    os.makedirs("data", exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")
    print(f"[✓] Extracted {len(chunks)} chunks from {pdf_path}")
