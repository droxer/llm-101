from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="./tokenization")
set_seed(42)
txt = generator("五竹", max_length=10)
print(txt)
