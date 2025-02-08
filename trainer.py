from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

dataset_path = "./data/shakespeare.txt"
model_dir = "./models"
output_dir = "./outputs"

tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
tokenizer.add_special_tokens(
    {
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
    }
)

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=dataset_path,
    block_size=32,
)
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments


training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=16,
    save_steps=2000,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
model.save_pretrained(model_dir)
