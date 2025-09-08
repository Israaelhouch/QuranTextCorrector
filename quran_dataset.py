import random
import json

with open("text_corrector/quran.txt", "r", encoding="utf-8") as f:
    ayat = f.read().split("\n")

print(ayat[0])
print(ayat[1])
def corrupt_text(text):
    # Randomly remove a letter or replace it
    letters = list(text)
    if len(letters) > 3:
        idx = random.randint(0, len(letters)-1)
        letters[idx] = random.choice("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
    # Randomly remove diacritics (simplified)
    diacritics = "ًٌٍَُِّْ"
    text_no_diac = ''.join([c for c in ''.join(letters) if c not in diacritics])
    return text_no_diac

corrupted_pairs = [(corrupt_text(a), a) for a in ayat]

with open("text_corrector/corrupted_quran_pairs.json", "w", encoding="utf-8") as f:
    json.dump(corrupted_pairs, f, ensure_ascii=False, indent=4)

print("Saved corrupted Quran text pairs to corrupted_quran_pairs.json")
