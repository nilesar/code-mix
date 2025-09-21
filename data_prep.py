# data_prep.py
import random
from datasets import load_dataset, Dataset
import json
from tqdm import tqdm

def simple_align_and_codemix(parallel_ds, mix_prob=0.4):
    """
    Very simple rule:
    - For each source-target pair, pick token positions to replace in source with target tokens (word-level)
    - Uses whitespace split — later replace with BPE-aware method (for simplicity).
    """
    outputs = []
    for item in tqdm(parallel_ds):
        en = item['translation']['en'].strip()
        hi = item['translation']['hi'].strip()
        en_tokens = en.split()
        hi_tokens = hi.split()
        # naive mapping: if lengths differ, align first min length tokens
        L = min(len(en_tokens), len(hi_tokens))
        codemix = []
        for i in range(len(en_tokens)):
            if i < L and random.random() < mix_prob:
                codemix.append(hi_tokens[i])  # inject target token
            else:
                codemix.append(en_tokens[i])
        outputs.append({"en": en, "hi": hi, "codemix": " ".join(codemix)})
    return outputs

def generate_and_save(dataset_name="cfilt/iitb-english-hindi", save_path="synthetic_codemix.jsonl", max_examples=50000):
    ds = load_dataset(dataset_name, split="train")
    ds = ds.select(range(min(len(ds), max_examples)))
    out = simple_align_and_codemix(ds)
    with open(save_path, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("Saved", len(out), "examples to", save_path)

if __name__ == "__main__":
    generate_and_save()
