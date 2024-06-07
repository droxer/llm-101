from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import GPT2TokenizerFast


tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = Sequence([NFKC()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
trainer = BpeTrainer(
    vocab_size=50000,
    show_progress=True,
    inital_alphabet=ByteLevel.alphabet(),
    special_tokens=special_tokens,
)

files = ["data/庆余年.txt"]

tokenizer.train(files, trainer)

newtokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
newtokenizer.save_pretrained("./tokenization")
