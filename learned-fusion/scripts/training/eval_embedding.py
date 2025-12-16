import glob
import re
from pathlib import Path

import numpy as np
import torch
from parser import parse_graph
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ======================
# CONFIG
# ======================
DATA_GLOB = "./data/torch_compile_out/*/my_ir_test_file_*.txt"
MODEL_DIR = "./models/embedding_model"
MAX_LEN = 512
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKEN_DROPOUT_PROB = 0.0  # must be 0 for evaluation
TOP_K = [1, 5, 10]

# ======================
# Preprocessing
# ======================
NUM_RE = re.compile(r"\b\d+\b")
SNAKE_RE = re.compile(r"_+")


def preprocess_data(text: str) -> str:
    text = NUM_RE.sub("<NUM>", text)
    text = SNAKE_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# Load all files, parse them, and collect node data.
def load_all_data():
    node_data_lst = set()
    file_paths = sorted(glob.glob(DATA_GLOB))

    if not file_paths:
        raise RuntimeError(f"No files matched {DATA_GLOB}")

    for path in file_paths:
        text = Path(path).read_text()
        nodes = parse_graph(text)

        for node in nodes.values():
            data = node["data"].strip()
            if data:
                node_data_lst.add(preprocess_data(data))

    node_data_lst = list(node_data_lst)

    print(f"Loaded {len(node_data_lst)} node data from {len(file_paths)} files")
    return node_data_lst


# ======================
# Dataset
# ======================
class NodeDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


# ======================
# Model wrapper
# ======================
class Embedder(torch.nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_embeddings = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(pooled, dim=-1)


# ======================
# Encoding
# ======================
@torch.no_grad()
def encode(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    all_embs = []
    for batch in tqdm(loader, desc="Encoding"):
        emb = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        )
        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0)


# ======================
# Paired Recall@K
# ======================
def paired_recall_at_k(embs1, embs2, k):
    sim = embs1 @ embs2.T
    topk = sim.topk(k, dim=1).indices
    targets = torch.arange(embs1.size(0)).unsqueeze(1)
    hits = (topk == targets).any(dim=1)
    return hits.float().mean().item()


# ======================
# Main
# ======================
def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = Embedder(MODEL_DIR).to(DEVICE)

    print("Loading data...")
    data = list(np.random.choice(load_all_data(), size=1_000, replace=False))
    dataset = NodeDataset(data, tokenizer)

    # Encode twice (deterministic views)
    print("Encoding data...")
    embs1 = encode(model, dataset)
    embs2 = encode(model, dataset)

    print("\nPaired Recall@K:")
    for k in TOP_K:
        r = paired_recall_at_k(embs1, embs2, k)
        print(f"Recall@{k}: {r:.4f}")


if __name__ == "__main__":
    main()
