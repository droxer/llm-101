from transformers import pipeline, set_seed

generator = pipeline(
    "text-generation",
    model="./models",
    device="mps",
)
set_seed(42)
txt = generator("to be or not to be", max_length=40)
print(txt)
