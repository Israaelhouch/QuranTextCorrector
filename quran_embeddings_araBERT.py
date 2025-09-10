from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import faiss
import numpy as np

# Load BERT Arabic model
model_name = "aubmindlab/bert-base-arabertv2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()  # evaluation mode


def encode_ayah(ayah_text):
    # Tokenize
    inputs = tokenizer(ayah_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)  # outputs.last_hidden_state shape: [1, seq_len, 768]
    token_embeddings = outputs.last_hidden_state[0]  # [seq_len, 768]
    # Mean pooling
    ayah_embedding = token_embeddings.mean(dim=0)
    return ayah_embedding.numpy()


# Load your normalized dataset
df = pd.read_csv("quran_dataset.csv")

# Encode all ayat
embeddings = []
for ayah in df["normalized_text"]:
    vec = encode_ayah(ayah)
    embeddings.append(vec)
embeddings = np.stack(embeddings).astype("float32")

# Build FAISS index
dimension = embeddings.shape[1]  # 768
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index & embeddings
faiss.write_index(index, "quran_index_araBERT.faiss")
np.save("quran_embeddings_araBERT.npy", embeddings)
df.to_csv("quran_dataset_with_embeddings.csv", index=False)
print("✅ FAISS index built with", embeddings.shape[0], "ayahs")

