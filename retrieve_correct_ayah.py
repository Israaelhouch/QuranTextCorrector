#pip install torch transformers faiss-cpu python-Levenshtein


import pandas as pd
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import Levenshtein
from utils import normalize_arabic, encode_text, retrieve_ayah


def highlight_errors(transcribed_text, correct_text):
    # Compute Levenshtein diff (character-level)
    ratio = Levenshtein.ratio(transcribed_text, correct_text)
    return ratio, correct_text  # ratio = similarity score


# Load dataset
df = pd.read_csv("quran_dataset.csv")

# Load embeddings
embeddings = np.load("quran_embeddings_araBERT.npy").astype("float32")

# Load FAISS index
dimension = embeddings.shape[1]
index = faiss.read_index("quran_index_araBERT.faiss")

# Load AraBERT
model_name = "aubmindlab/bert-base-arabertv2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print(model.eval())

# Example: simulate chunked transcription
chunks = [
    "ذَلِكَ الْكِتَابُ لَ",               # partial
    "ذَلِكَ الْكِتَابُ لَا رَيْبَ",       # more complete
    "ذَلِكَ الْكِتَابُ لَا رَيْبَ فِيهِ",  # correct
    "ذَلِكَ الْكِتَابُ لَا رَيْبَ فِيهِ هُدَى"  # missing end
]

threshold = 0.85  # similarity threshold

for chunk in chunks:
    norm_chunk = normalize_arabic(chunk)
    vec = encode_text(norm_chunk)
    dist, idx = retrieve_ayah(vec)
    correct_ayah = normalize_arabic(df.iloc[idx]["ayah_text"])
    
    # Levenshtein similarity
    score, feedback = highlight_errors(norm_chunk, correct_ayah)
    
    # Check threshold
    if score < threshold:
        print(f"⚠️ Mistake detected!")
        print(f"Transcribed chunk: {chunk}")
        print(f"Closest ayah  : {df.iloc[idx]['ayah_text']}")
        print(f"Similarity    : {score:.2f}\n")
    else:
        print(f"✅ Chunk correct (Similarity {score:.2f})")

