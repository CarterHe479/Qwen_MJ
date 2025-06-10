import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            texts.append(text.strip())
    return "\n".join(texts)

def split_text(text, max_chunk_len=300):
    chunks = []
    paragraph = ""
    for line in text.split("\n"):
        if len(paragraph) + len(line) < max_chunk_len:
            paragraph += line.strip() + " "
        else:
            chunks.append(paragraph.strip())
            paragraph = line.strip() + " "
    if paragraph:
        chunks.append(paragraph.strip())
    return chunks

def convert_pdf_to_chunks(pdf_path, output_txt="data/kb_docs.txt"):
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    os.makedirs("data", exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")
    print(f"[✓] Extracted {len(chunks)} chunks from {pdf_path}")
