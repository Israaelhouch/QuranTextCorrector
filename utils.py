# Normalize Arabic text (remove diacritics, unify letters)
import re
def remove_diacritics(text):
    arabic_diacritics = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    return re.sub(arabic_diacritics, "", text)

def normalize_arabic(text):
    text = remove_diacritics(text)
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة","ه")
    text = ' '.join(text.split())
    return text

# Encode a chunk into vector
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state[0]  # [seq_len, 768]
    return token_embeddings.mean(dim=0).numpy()

# Retrieve nearest ayah from FAISS
def retrieve_ayah(chunk_vec, k=1):
    D, I = index.search(np.array([chunk_vec]).astype("float32"), k)
    return D[0][0], I[0][0]  # distance, ayah index
