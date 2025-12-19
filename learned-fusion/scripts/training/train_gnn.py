import glob
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi
from parser import parse_ir_file, parse_score_file, preprocess_data
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DATA_ROOT = "./data/torch_compile_out"
EMBEDDING_MODEL_DIR = "./models/embedding_model"
OUTPUT_DIR = "./output/training"
CHECKPOINT_DIR = "./models/gnn_checkpoints"
MODEL_DIR = "./models/gnn_model"

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
EPOCHS = 20
LR = 3e-4
PAIRWISE_MARGIN = 5.0
EQUAL_SCORE_REG = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = None  # set to None to use all data

EMBED_DIM = 768
EDGE_FEAT_DIM = 4

HF_REPO_ID = "sameer-n012/cs521-gnn"

SEED = 123


def load_all_graphs(tokenizer, encoder) -> list[Data]:
    graphs = []

    pairs = []
    for model_dir in Path(DATA_ROOT).iterdir():
        for ir_path in model_dir.glob("my_ir_test_file_*.txt"):
            idx = ir_path.stem.split("_")[-1]
            score_path = model_dir / f"my_score_fusions_test_file_{idx}.txt"
            if score_path.exists():
                pairs.append((ir_path, score_path))

    if SAMPLE_SIZE and len(pairs) > SAMPLE_SIZE:
        pairs = random.sample(pairs, SAMPLE_SIZE)

    for ir_path, score_path in tqdm(pairs, desc="Loading graphs"):
        graph = build_graph(
            ir_path.read_text(), score_path.read_text(), tokenizer, encoder
        )
        if graph is not None:
            graphs.append(graph)

    return graphs


def load_node_data(text: str):
    nodes = parse_ir_file(text)
    for n in nodes.keys():
        if nodes[n]["data"] is not None:
            nodes[n]["data"] = preprocess_data(nodes[n]["data"])
    return nodes


def embed_node_data(nodes, tokenizer, encoder):
    names = sorted(nodes.keys())

    x = []
    for n in tqdm(names, desc="Embedding node data"):
        # hash data to use as file name
        data_hash = hashlib.md5(nodes[n]["data"].encode("utf-8")).hexdigest()

        cache_dir = Path("./node_embedding_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{n}_{data_hash}.pt"

        if cache_path.exists():
            e = torch.load(cache_path)
            x.append(e.to(DEVICE))
            continue

        e = embed_text(tokenizer, encoder, nodes[n]["data"])

        # cache embedding on disk
        torch.save(e.detach().cpu(), cache_path)

        x.append(e.to(DEVICE))

    return torch.stack(x)


def tuple_gt(a, b):
    """
    Returns True where a > b lexicographically, element-wise.
    a, b: tensors of shape (E, T)
    """

    assert a.shape == b.shape
    *leading, T = a.shape
    # Start with all False
    result = torch.zeros(leading, dtype=torch.bool, device=a.device)
    # Also keep track of positions where we’ve already decided
    decided = torch.zeros_like(result)

    for t in range(T):
        gt = a[..., t] > b[..., t]
        lt = a[..., t] < b[..., t]

        # Only update where not decided
        result = torch.where(~decided & gt, torch.ones_like(result), result)
        decided = decided | gt | lt  # decided wherever a != b at this position

    return result


def build_graph(ir_text: str, score_text: str, tokenizer, encoder) -> Data | None:
    nodes = load_node_data(ir_text)
    scores = parse_score_file(score_text)

    names = sorted(nodes.keys())
    idx = {n: i for i, n in enumerate(names)}

    x = embed_node_data(nodes, tokenizer, encoder)

    edge_index, edge_attr, edge_label = [], [], []
    valid = []

    for src, node in nodes.items():
        # for dst in node["users"]:
        for dst, _ in nodes.items():
            if src == dst:
                continue

            # keep edges with scores + edges to users
            if (src, dst) not in scores and dst not in node["users"]:
                continue

            if dst not in idx:
                continue

            edge_index.append([idx[src], idx[dst]])
            label = scores.get((src, dst), (-1, False, -1, -1))

            edge_attr.append(
                torch.tensor(
                    [
                        float(label[0]),
                        float(label[1]),
                        float(label[2]),
                        float(label[3]),
                    ]
                )
            )

            # ranked_label = label_to_rank_value(label)
            edge_label.append(label)
            valid.append(label != (-1, False, -1, -1))

    if len(edge_index) == 0:
        return None

    y = torch.tensor(edge_label, dtype=torch.float)
    valid = torch.tensor(valid, dtype=torch.bool)
    # valid = y != (-1, False, -1, -1)
    if valid.any():
        y_valid = y[valid]
        # y[valid] = (y_valid - y_valid.mean()) / (y_valid.std() + 1e-6)
        y_mean = y_valid.mean(dim=0, keepdim=True)  # mean per feature
        y_std = y_valid.std(dim=0, unbiased=False, keepdim=True)  # std per feature
        y_std[y_std < 1e-6] = 1.0

        y[valid] = (y_valid - y_mean) / y_std

    return Data(
        x=x,
        edge_index=torch.tensor(edge_index).T,
        edge_attr=torch.stack(edge_attr),
        y=y,
    )


def train_val_split(graphs, val_frac=0.1):
    random.shuffle(graphs)
    n_val = int(len(graphs) * val_frac)
    return graphs[n_val:], graphs[:n_val]


@torch.no_grad()
def embed_text(tokenizer, encoder, text: str) -> torch.Tensor:
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    ).to(DEVICE)

    out = encoder(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    pooled = (out * mask).sum(1) / mask.sum(1)
    return F.normalize(pooled.squeeze(0), dim=-1).cpu()


class EdgeGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn1 = GATv2Conv(EMBED_DIM, 256, heads=4)
        self.gnn2 = GATv2Conv(256 * 4, 256)

        self.edge_mlp = nn.Sequential(
            nn.Linear(256 * 2 + EDGE_FEAT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, data):
        h = F.relu(self.gnn1(data.x, data.edge_index))
        h = F.relu(self.gnn2(h, data.edge_index))

        src, dst = data.edge_index
        edge_input = torch.cat([h[src], h[dst], data.edge_attr], dim=1)
        pred = self.edge_mlp(edge_input).squeeze(-1)
        # pred = torch.tanh(pred)
        return pred


def pairwise_ranking_loss(pred, target):
    """
    pred:   (E,)
    target: (E,)
    """

    # valid = target != -1
    # pred = pred[valid]
    # target = target[valid]

    # target: (E, 4)
    invalid_mask = (
        (target[:, 0] == -1)
        & (target[:, 1] == 0)
        & (target[:, 2] == -1)
        & (target[:, 3] == -1)
    )

    # keep only valid edges
    valid = ~invalid_mask

    pred = pred[valid]  # (E_valid,)
    target = target[valid, :]  # (E_valid, 4)

    n = pred.numel()
    if n < 2:
        return pred.sum() * 0.0

    # pairwise differences
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)

    # --------------------------------------------------
    # Discount pairs involving "unknown" edges (target == -1)
    # --------------------------------------------------
    # valid_i = target.unsqueeze(1) != -1
    # valid_j = target.unsqueeze(0) != -1
    # pair_weight = (valid_i & valid_j).float()

    # Heavily discount (but not zero, for stability)
    # pair_weight = pair_weight * 1.0 + (1.0 - pair_weight) * 0.05
    # 0.05 can be smaller (0.01) if you want near-ignore

    # # --------------------------------------------------
    # # Ranking loss
    # # --------------------------------------------------
    # rank_mask = target_diff > 0
    # rank_loss = F.relu(PAIRWISE_MARGIN - pred_diff)

    # # --------------------------------------------------
    # # Tie loss
    # # --------------------------------------------------
    # tie_mask = target_diff == 0
    # tie_loss = pred_diff**2

    # loss = 0.0

    # if rank_mask.any():
    #     loss = loss + (rank_loss[rank_mask] * pair_weight[rank_mask]).mean()

    # if tie_mask.any():
    #     loss = (
    #         loss + EQUAL_SCORE_REG * (tie_loss[tie_mask] * pair_weight[tie_mask]).mean()
    #     )

    # loss = loss - 0.01 * pred.var(unbiased=False)

    # return loss

    target_diff = tuple_gt(
        target.unsqueeze(1).expand(-1, n, -1), target.unsqueeze(0).expand(n, -1, -1)
    )  # (E, E) bool

    # pred_diff = torch.clamp(pred_diff, -5.0, 5.0)

    # rank loss
    rank_loss = F.relu(PAIRWISE_MARGIN - pred_diff)
    loss = rank_loss[target_diff].mean() if target_diff.any() else 0.0

    # tie loss
    tie_mask = ~target_diff & (target.unsqueeze(1) == target.unsqueeze(0)).all(dim=2)
    tie_loss = (pred_diff**2)[tie_mask].mean() if tie_mask.any() else 0.0

    loss += EQUAL_SCORE_REG * tie_loss

    # variance regularizer to prevent collapse
    # loss += 0.01 * pred.var(unbiased=False)
    # loss += 1e-3 * (pred**2).mean()

    return loss


def train(train_graphs, val_graphs, results_file=None, batch_size=BATCH_SIZE):
    loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    model = EdgeGNN().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    best_val_corr = -1.0
    best_epoch = -1
    best_ckpt_path = None

    results = {
        "config": {
            "device": DEVICE,
            "batch_size": batch_size,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": batch_size * GRAD_ACCUM_STEPS,
            "epochs": EPOCHS,
            "lr": LR,
            "embed_dim": EMBED_DIM,
            "edge_feat_dim": EDGE_FEAT_DIM,
            "equal_score_reg": EQUAL_SCORE_REG,
            "pairwise_margin": PAIRWISE_MARGIN,
            "hf_repo_id": HF_REPO_ID,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_size": len(train_graphs),
            "completed": False,
        },
        "epochs": [],
    }

    if results_file:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        pbar.set_postfix(loss=f"{total_loss:.4f}")

        for step, batch in enumerate(pbar):
            batch = batch.to(DEVICE)
            pred = model(batch)

            if pred.numel() < 2:
                continue

            loss = pairwise_ranking_loss(pred, batch.y)

            # optimizer.zero_grad()
            loss /= GRAD_ACCUM_STEPS
            loss.backward()
            # optimizer.step()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            pbar.set_postfix(loss=f"{(total_loss / (step + 1)):.4f}")

            if step % 50 == 0:
                with torch.no_grad():
                    print(
                        f"pred std: {pred.std().item():.4f}, "
                        f"min: {pred.min().item():.3f}, "
                        f"max: {pred.max().item():.3f}"
                        f"loss: {total_loss / (step + 1):.4f}"
                    )

        optimizer.step()
        optimizer.zero_grad()

        avg_loss = total_loss / len(loader)
        val_corr, val_acc = validate(model, val_graphs)

        print(
            f"Epoch {epoch + 1} loss: {avg_loss:.4f}, Val Spearman: {val_corr:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_epoch = epoch + 1
            best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best.pt")
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                {"loss": avg_loss, "val_corr": val_corr, "val_acc": val_acc},
                best_ckpt_path,
            )

        results["epochs"].append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "val_corr": val_corr,
                "val_acc": val_acc,
                "steps": len(loader),
            }
        )

        if results_file:
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

    results["config"]["completed"] = True
    results["best"] = {
        "epoch": best_epoch,
        "best_val_corr": best_val_corr,
    }

    if results_file:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    # Load best model before returning
    if best_ckpt_path:
        load_checkpoint(model, optimizer=None, path=best_ckpt_path, device=DEVICE)

    return model


@torch.no_grad()
def validate(model, val_graphs):
    return _validate_corr(model, val_graphs), _validate_acc(model, val_graphs)
    # return _validate_corr(model, val_graphs), 0.0


@torch.no_grad()
def _validate_acc(model, val_graphs):
    """
    Compute pairwise accuracy over all edges in validation graphs.
    Accuracy = fraction of correctly ranked pairs according to tuple_gt.
    """
    model.eval()
    total_acc = 0.0
    total_graphs = 0

    for data in val_graphs:
        data = data.to(DEVICE)
        pred = model(data)  # (E,)
        target = data.y  # (E, T)

        E = pred.shape[0]
        if E < 2:
            continue

        # Compute all pairwise comparisons at once
        # target_diff[i,j] = True if target[i] > target[j]
        target_gt = tuple_gt(
            target.unsqueeze(1).expand(-1, E, -1), target.unsqueeze(0).expand(E, -1, -1)
        )  # shape (E, E), bool

        # pred_gt[i,j] = True if pred[i] > pred[j]
        pred_gt = pred.unsqueeze(1) > pred.unsqueeze(0)  # shape (E, E), bool

        # Only consider pairs where target_gt is True
        correct_pairs = (pred_gt & target_gt).sum().item()
        total_pairs = target_gt.sum().item()

        if total_pairs > 0:
            total_acc += correct_pairs / total_pairs
            total_graphs += 1

    model.train()
    return total_acc / max(total_graphs, 1)


@torch.no_grad()
def _validate_corr(model, val_graphs, device=DEVICE):
    """
    Compute average Spearman rank correlation across graphs.
    """
    model.eval()
    correlations = []

    with torch.no_grad():
        for data in val_graphs:
            data = data.to(device)
            pred = model(data)
            target = data.y

            if pred.numel() < 2:
                continue

            # Convert to numpy for Spearman correlation
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()

            corr, _ = spearmanr(pred_np, target_np)
            if not np.isnan(corr).any():
                correlations.append(corr)

    model.train()
    if len(correlations) == 0:
        return 0.0
    return float(np.mean(correlations))


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


@torch.no_grad()
def infer_single_graph(ir_path, score_path, ckpt_path, tokenizer, encoder):
    model = EdgeGNN().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    graph = build_graph(ir_path.read_text(), score_path.read_text(), tokenizer, encoder)
    if graph is None:
        return None
    return model(graph.to(DEVICE)).cpu()


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_DIR)
    encoder = AutoModel.from_pretrained(EMBEDDING_MODEL_DIR).to(DEVICE)
    encoder.eval()

    graphs = load_all_graphs(tokenizer, encoder)
    print(graphs[0])
    train_graphs, val_graphs = train_val_split(graphs)

    train_out = os.path.join(OUTPUT_DIR, "gnn_training_results.json")
    model = train(train_graphs, val_graphs, results_file=train_out)

    # save model locally
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_class": "EdgeGNN",
            "embed_dim": EMBED_DIM,
            "edge_feat_dim": EDGE_FEAT_DIM,
        },
        os.path.join(MODEL_DIR, "model.pt"),
    )
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
