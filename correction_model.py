import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

with open("corrupted_quran_pairs.json", "r", encoding="utf-8") as f:
    corrupted_pairs = json.load(f)

dataset_dict = {
    "input_text": [c for c, t in corrupted_pairs],
    "target_text": [t for c, t in corrupted_pairs]
}

dataset = Dataset.from_dict(dataset_dict)

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess(batch):
    inputs = tokenizer(batch["input_text"], max_length=64, truncation=True, padding="max_length")
    labels = tokenizer(batch["target_text"], max_length=64, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True)


training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-5,
    save_strategy="no",
    logging_steps=10,
    no_cuda=True, 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("quran_corrector_model")
tokenizer.save_pretrained("quran_corrector_model")
