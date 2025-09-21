# train_tokenizer.py
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import os

def collect_sentences(dataset_name="cfilt/iitb-english-hindi", split="train", lang_keys=("en","hi")):
    ds = load_dataset(dataset_name, split=split)
    for item in ds:
        tr = item['translation']
        yield tr.get(lang_keys[0], "")
        yield tr.get(lang_keys[1], "")

def train_tokenizer(out_dir="bpe_tokenizer", vocab_size=32000):
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(collect_sentences(), vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=["<s>", "</s>", "<pad>", "<unk>"])
    tokenizer.save_model(out_dir)
    print("Saved tokenizer to", out_dir)

if __name__ == "__main__":
    train_tokenizer()
