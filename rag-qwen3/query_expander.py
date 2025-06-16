from sentence_transformers import SentenceTransformer, util

paraphraser = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def expand_query(query):
    paraphrases = util.paraphrase_mining(paraphraser, [query])
    return [p[1] for p in paraphrases[:3]]
