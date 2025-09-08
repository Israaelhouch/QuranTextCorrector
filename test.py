from transformers import T5ForConditionalGeneration, T5Tokenizer

model_path = "quran_corrector_model"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto")

device = "mps"
model.to(device)

input_text = "السلام علينم، كيف حالكك؟"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_length=128,
    num_beams=4,
    early_stopping=True
)

corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("input text:", input_text)
print("Corrected Text:", corrected_text)
