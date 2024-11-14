import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

dataset = pd.read_csv('insurance_claims.csv')

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    save_steps=10_00,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {eval_results['eval_loss']:.2f}")
