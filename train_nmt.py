import torch
from torch.utils.data import DataLoader
from dataset import CodeMixDataset, collate_fn
from model import CodeMixModel
import os
from tqdm import tqdm

def train_nmt(tokenizer_dir="bpe_tokenizer", parallel_jsonl="synthetic_codemix.jsonl", epochs=5, batch_size=16, save_path="nmt_checkpoint.pt"):
    ds = CodeMixDataset(parallel_jsonl, tokenizer_dir=tokenizer_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    vocab_size = 32000  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CodeMixModel(tokenizer_vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.NLLLoss(ignore_index=1)  
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader):
            src = batch['src_ids'].to(device)
            tgt = batch['tgt_ids'].to(device)
            pos = batch['pos_ids'].to(device)
            ner = batch['ner_ids'].to(device)
            optimizer.zero_grad()
            outputs = model(src, pos, ner, tgt)
            logp = torch.log(outputs + 1e-12)
            loss = criterion(logp.view(-1, logp.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {ep} loss {total_loss/len(loader)}")
        torch.save(model.state_dict(), save_path)
    print("Saved NMT model to", save_path)

if __name__ == "__main__":
    train_nmt()
