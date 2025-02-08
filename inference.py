from transformers import pipeline, GPT2Tokenizer, set_seed

model_dir = "./models"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

generator = pipeline(
    "text-generation",
    model=model_dir,   
    device="mps",
)
set_seed(42)
txt = generator("to be or not to be", max_length=30, truncation=True, pad_token_id=tokenizer.eos_token_id)
print(txt)
