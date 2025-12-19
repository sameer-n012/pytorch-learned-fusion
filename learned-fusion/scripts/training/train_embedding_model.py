import glob
import itertools
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from parser import parse_ir_file, preprocess_data
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DATA_GLOB = "./data/torch_compile_out/*/my_ir_test_file_*.txt"
MODEL_DIR = "./models/embedding_model"
OUTPUT_DIR = "./output/training"
CHECKPOINT_DIR = "./models/checkpoints"
MODEL_NAME = "microsoft/codebert-base"
MAX_LEN = 512
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4
LOSS_TEMPERATURE = 0.1
EPOCHS = 5
LR = 3e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROPOUT_PROB = 0.2
SAMPLE_SIZE = None  # set to None to use all data
TOKEN_DROPOUT_PROB = 0.15


HF_REPO_ID = "sameer-n012/cs521-embedding-model"


# Load all files, parse them, and collect node data.
def load_all_data():
    node_data_lst = set()
    file_paths = sorted(glob.glob(DATA_GLOB))

    if not file_paths:
        raise RuntimeError(f"No files matched {DATA_GLOB}")

    for path in file_paths:
        text = Path(path).read_text()
        nodes = parse_ir_file(text)

        for node in nodes.values():
            data = node["data"].strip()
            if data:
                node_data_lst.add(preprocess_data(data))

    node_data_lst = list(node_data_lst)

    # randomly sample SAMPLE_SIZE nodes
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(node_data_lst):
        node_data_lst = list(
            np.random.choice(node_data_lst, size=SAMPLE_SIZE, replace=False)
        )
    print(f"Loaded {len(node_data_lst)} node data from {len(file_paths)} files")
    return node_data_lst


def train_val_split(data, val_frac=0.05):
    random.seed(123)
    data = data.copy()
    random.shuffle(data)
    n_val = int(len(data) * val_frac)
    return data[n_val:], data[:n_val]


class NodeDataset(Dataset):
    def __init__(self, text, tokenizer, token_dropout_prob=TOKEN_DROPOUT_PROB):
        self.texts = text
        self.tokenizer = tokenizer
        self.token_dropout_prob = token_dropout_prob

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

        # two independent views
        ids1 = self.token_dropout(
            input_ids,
            self.tokenizer.mask_token_id,
        )
        ids2 = self.token_dropout(
            input_ids,
            self.tokenizer.mask_token_id,
        )

        return {
            "input_ids_1": ids1,
            "attention_mask_1": attention_mask,
            "input_ids_2": ids2,
            "attention_mask_2": attention_mask,
        }

    def token_dropout(self, input_ids, mask_token_id):
        if self.token_dropout_prob <= 0:
            return input_ids

        # never mask special tokens
        special_mask = (
            (input_ids == mask_token_id)
            | (input_ids == 0)  # pad
            | (input_ids == 101)  # [CLS]
            | (input_ids == 102)  # [SEP]
        )

        dropout_mask = (
            torch.rand(input_ids.shape) < self.token_dropout_prob
        ) & ~special_mask
        input_ids = input_ids.clone()
        input_ids[dropout_mask] = mask_token_id
        return input_ids


class Embedder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            hidden_dropout_prob=DROPOUT_PROB,
            attention_probs_dropout_prob=DROPOUT_PROB,
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_embeddings = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(1) / mask.sum(1)
        return nn.functional.normalize(pooled, dim=-1)


def simcse_loss(embeddings, labels, temperature):
    embeddings = F.normalize(embeddings, dim=1)
    sim = embeddings @ embeddings.T / temperature
    sim.fill_diagonal_(-1e9)
    return F.cross_entropy(sim, labels)


# train model
def train(train_data, val_data, results_file=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = NodeDataset(train_data, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    val_dataset = NodeDataset(
        val_data,
        tokenizer,
        token_dropout_prob=0.0,
    )

    model = Embedder(MODEL_NAME).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    best_r1 = -1.0
    best_epoch = -1
    best_ckpt_path = None

    results = {
        "config": {
            "model": MODEL_NAME,
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "temperature": LOSS_TEMPERATURE,
            "epochs": EPOCHS,
            "lr": LR,
            "dropout_prob": DROPOUT_PROB,
            "token_dropout_prob": TOKEN_DROPOUT_PROB,
            "max_len": MAX_LEN,
            "hf_repo_id": HF_REPO_ID,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_size": len(dataset),
            "completed": False,
        },
        "epochs": [],
    }

    if results_file:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    model.train()
    for epoch in range(EPOCHS):
        ts = time.time()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        pbar.set_postfix(loss=f"{total_loss:.4f}")
        for step, batch in enumerate(pbar):
            emb1 = model(
                batch["input_ids_1"].to(DEVICE),
                batch["attention_mask_1"].to(DEVICE),
            )
            emb2 = model(
                batch["input_ids_2"].to(DEVICE),
                batch["attention_mask_2"].to(DEVICE),
            )

            embeddings = torch.cat([emb1, emb2], dim=0)
            labels = torch.arange(emb1.size(0), device=DEVICE)
            labels = torch.cat([labels + emb1.size(0), labels], dim=0)

            loss = simcse_loss(embeddings, labels, LOSS_TEMPERATURE)

            loss /= GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            pbar.set_postfix(loss=f"{total_loss / (step + 1):.4f}")

        optimizer.step()
        optimizer.zero_grad()

        avg_loss = total_loss / len(loader)
        te = time.time() - ts
        print(f"Epoch {epoch + 1} loss: {avg_loss:.4f} ({te:.2f}s)")

        embs1, embs2 = encode_pairs(model, val_dataset)
        r1 = paired_recall_at_k(embs1, embs2, k=1)
        r5 = paired_recall_at_k(embs1, embs2, k=5)

        print(f"Val Recall@1: {r1:.4f}, Recall@5: {r5:.4f}")

        is_best = r1 > best_r1
        if is_best:
            best_r1 = r1
            best_epoch = epoch + 1
            best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best.pt")

        save_checkpoint(
            model,
            optimizer,
            epoch=epoch + 1,
            metrics={"recall@1": r1, "recall@5": r5},
            path=os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}.pt"),
        )

        if is_best:
            save_checkpoint(
                model,
                optimizer,
                epoch=epoch + 1,
                metrics={"recall@1": r1, "recall@5": r5},
                path=best_ckpt_path,
            )

        results["epochs"].append(
            {
                "epoch": epoch + 1,
                "loss": round(avg_loss, 6),
                "val_recall_at_1": round(r1, 6),
                "val_recall_at_5": round(r5, 6),
                "steps": len(loader),
                "time": te,
            }
        )

        if results_file:
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

    results["config"]["completed"] = True
    results["best"] = {
        "epoch": best_epoch,
        "val_recall_at_1": round(best_r1, 6),
    }

    if results_file:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Training results written to {results_file}")

    if best_ckpt_path:
        load_checkpoint(
            model,
            optimizer=None,
            path=os.path.join(CHECKPOINT_DIR, "best.pt"),
            device=DEVICE,
        )

    return model, tokenizer


@torch.no_grad()
def encode_pairs(model, dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    embs1, embs2 = [], []

    for batch in loader:
        e1 = model(
            batch["input_ids_1"].to(DEVICE),
            batch["attention_mask_1"].to(DEVICE),
        )
        e2 = model(
            batch["input_ids_2"].to(DEVICE),
            batch["attention_mask_2"].to(DEVICE),
        )

        embs1.append(e1.cpu())
        embs2.append(e2.cpu())

    return torch.cat(embs1), torch.cat(embs2)


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def paired_recall_at_k(embs1, embs2, k=5):
    sim = embs1 @ embs2.T  # (N, N)

    topk = sim.topk(k, dim=1).indices
    targets = torch.arange(embs1.size(0)).unsqueeze(1)

    hits = (topk == targets).any(dim=1)
    return hits.float().mean().item()


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    train_out_file = os.path.join(OUTPUT_DIR, "embedding_training_results.json")

    node_data = load_all_data()
    train_data, val_data = train_val_split(node_data, val_frac=0.05)
    model, tokenizer = train(train_data, val_data, results_file=train_out_file)

    # save model locally
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")

    # upload to hf
    api = HfApi()
    api.create_repo(
        repo_id=HF_REPO_ID,
        private=False,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=HF_REPO_ID,
        repo_type="model",
    )

    print(f"Model uploaded to https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
