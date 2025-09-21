# # dataset.py
# import os
# import json
# import torch
# from torch.utils.data import Dataset
# from tokenizers import ByteLevelBPETokenizer
# import spacy
# from collections import defaultdict

# nlp = spacy.load("en_core_web_sm")

# class CodeMixDataset(Dataset):
#     def __init__(self, jsonl_path, tokenizer_dir="bpe_tokenizer", max_len=128):
#         self.samples = []
#         with open(jsonl_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 self.samples.append(json.loads(line))
#         self.tokenizer = ByteLevelBPETokenizer(os.path.join(tokenizer_dir, "vocab.json"),
#                                                os.path.join(tokenizer_dir, "merges.txt"))
#         self.max_len = max_len
#         # build small POS and NER dictionaries
#         self.pos2id = defaultdict(lambda: len(self.pos2id))
#         self.pos2id["<pad>"] = 0
#         self.ent2id = defaultdict(lambda: len(self.ent2id))
#         self.ent2id["<pad>"] = 0

#     def __len__(self):
#         return len(self.samples)

#     def sentence_to_tokens(self, s):
#         # tokenizer returns tokens and ids
#         enc = self.tokenizer.encode(s)
#         return enc.ids, enc.tokens

#     def linguistic_tags(self, s):
#         doc = nlp(s)
#         pos_ids = []
#         ner_ids = []
#         # map token-level via whitespace tokens; to keep consistent, use spaCy tokens and map to BPE token count in a naive way.
#         for tok in doc:
#             pos_ids.append(self.pos2id[tok.pos_])
#         # NER: mark tokens that are part of entities
#         ent_labels = ["O"] * len(doc)
#         for ent in doc.ents:
#             for i in range(ent.start, ent.end):
#                 ent_labels[i] = ent.label_
#         for lab in ent_labels:
#             ner_ids.append(self.ent2id[lab])
#         return pos_ids, ner_ids

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         en = sample['en']
#         codemix = sample['codemix']
#         # tokenization (ids)
#         src_ids, src_tokens = self.sentence_to_tokens(en)
#         tgt_ids, tgt_tokens = self.sentence_to_tokens(codemix)
#         # linguistic features from English (source)
#         pos_ids, ner_ids = self.linguistic_tags(en)
#         # pad/truncate
#         src_ids = src_ids[:self.max_len]
#         tgt_ids = tgt_ids[:self.max_len]
#         # convert lists to tensors (we'll collate to pad)
#         return {
#             'src_ids': torch.tensor(src_ids, dtype=torch.long),
#             'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long),
#             'pos_ids': torch.tensor(pos_ids, dtype=torch.long),
#             'ner_ids': torch.tensor(ner_ids, dtype=torch.long),
#             'src_text': en,
#             'tgt_text': codemix
#         }

# def collate_fn(batch):
#     # pad sequences
#     from torch.nn.utils.rnn import pad_sequence
#     srcs = [b['src_ids'] for b in batch]
#     tgts = [b['tgt_ids'] for b in batch]
#     pos = [b['pos_ids'] for b in batch]
#     ner = [b['ner_ids'] for b in batch]
#     src_p = pad_sequence(srcs, batch_first=True, padding_value=1)  # <pad> id might be 1 depending on tokenizer; adjust
#     tgt_p = pad_sequence(tgts, batch_first=True, padding_value=1)
#     pos_p = pad_sequence(pos, batch_first=True, padding_value=0)
#     ner_p = pad_sequence(ner, batch_first=True, padding_value=0)
#     return {
#         'src_ids': src_p,
#         'tgt_ids': tgt_p,
#         'pos_ids': pos_p,
#         'ner_ids': ner_p,
#         'raw': batch
#     }
import os
import json
import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
import spacy
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

class CodeMixDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer_dir="bpe_tokenizer", max_len=128):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = ByteLevelBPETokenizer(
            os.path.join(tokenizer_dir, "vocab.json"),
            os.path.join(tokenizer_dir, "merges.txt")
        )
        self.max_len = max_len
        # build small POS and NER dictionaries
        self.pos2id = defaultdict(lambda: len(self.pos2id))
        self.pos2id["<pad>"] = 0
        self.ent2id = defaultdict(lambda: len(self.ent2id))
        self.ent2id["<pad>"] = 0

    def __len__(self):
        return len(self.samples)

    def sentence_to_tokens(self, s):
        enc = self.tokenizer.encode(s)
        return enc.ids, enc.tokens

    def linguistic_tags(self, s, target_len):
        """
        Extract POS + NER and align/pad them to match target_len (src_ids length).
        """
        doc = nlp(s)
        pos_ids = [self.pos2id[tok.pos_] for tok in doc]

        # build NER sequence same length as doc
        ent_labels = ["O"] * len(doc)
        for ent in doc.ents:
            for i in range(ent.start, ent.end):
                ent_labels[i] = ent.label_
        ner_ids = [self.ent2id[lab] for lab in ent_labels]

        # pad/truncate to match src_ids length
        pos_ids = pos_ids[:target_len] + [0] * max(0, target_len - len(pos_ids))
        ner_ids = ner_ids[:target_len] + [0] * max(0, target_len - len(ner_ids))

        return pos_ids, ner_ids

    def __getitem__(self, idx):
        sample = self.samples[idx]
        en = sample['en']
        codemix = sample['codemix']

        # tokenize source and target
        src_ids, _ = self.sentence_to_tokens(en)
        tgt_ids, _ = self.sentence_to_tokens(codemix)

        # truncate to max_len
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]

        # extract aligned linguistic features
        pos_ids, ner_ids = self.linguistic_tags(en, len(src_ids))

        return {
            'src_ids': torch.tensor(src_ids, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long),
            'pos_ids': torch.tensor(pos_ids, dtype=torch.long),
            'ner_ids': torch.tensor(ner_ids, dtype=torch.long),
            'src_text': en,
            'tgt_text': codemix
        }


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    srcs = [b['src_ids'] for b in batch]
    tgts = [b['tgt_ids'] for b in batch]
    pos = [b['pos_ids'] for b in batch]
    ner = [b['ner_ids'] for b in batch]

    src_p = pad_sequence(srcs, batch_first=True, padding_value=1)  # tokenizer <pad> = 1
    tgt_p = pad_sequence(tgts, batch_first=True, padding_value=1)
    pos_p = pad_sequence(pos, batch_first=True, padding_value=0)   # pos <pad> = 0
    ner_p = pad_sequence(ner, batch_first=True, padding_value=0)   # ner <pad> = 0

    return {
        'src_ids': src_p,
        'tgt_ids': tgt_p,
        'pos_ids': pos_p,
        'ner_ids': ner_p,
        'raw': batch
    }
