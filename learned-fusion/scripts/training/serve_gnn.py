import os
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, Response, request
from huggingface_hub import snapshot_download
from parser import parse_ir_file, parse_score_file, preprocess_data
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv
from train_gnn import EdgeGNN, build_graph
from transformers import AutoModel, AutoTokenizer

HF_GNN_REPO = "sameer-n012/cs521-gnn"
HF_EMBED_REPO = "sameer-n012/cs521-embedding-model"

LOCAL_GNN_DIR = "./models/gnn_model"
LOCAL_EMBED_DIR = "./models/embedding_model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_model(repo_id: str, local_dir: str):
    """Download HF repo if local directory does not exist."""
    if not Path(local_dir).exists():
        print(f"Downloading {repo_id} → {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )


def load_models():
    ensure_model(HF_GNN_REPO, LOCAL_GNN_DIR)
    ensure_model(HF_EMBED_REPO, LOCAL_EMBED_DIR)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_EMBED_DIR)
    encoder = AutoModel.from_pretrained(LOCAL_EMBED_DIR).to(DEVICE)
    encoder.eval()

    ckpt = torch.load(
        os.path.join(LOCAL_GNN_DIR, "model.pt"),
        map_location=DEVICE,
    )

    gnn = EdgeGNN().to(DEVICE)
    gnn.load_state_dict(ckpt["model_state"])
    gnn.eval()

    return gnn, tokenizer, encoder


gnn, tokenizer, encoder = load_models()
app = Flask(__name__)


@app.route("/infer", methods=["POST"])
def infer():
    """
    Request body:
      <node_data>
      raw IR text
      </node_data>
      <edge_data>
      edge text
      </edge_data>

    Response body:
      score_fusions-style text
    """
    body = request.data.decode("utf-8")

    if not body.strip():
        return Response("ERROR: empty input\n", status=400, mimetype="text/plain")

    def extract_block(text, start_tag, end_tag):
        start = text.find(start_tag)
        end = text.find(end_tag)
        if start == -1 or end == -1 or end < start:
            return None
        return text[start + len(start_tag) : end].strip()

    ir_text = extract_block(body, "<node_data>", "</node_data>")
    edge_text = extract_block(body, "<edge_data>", "</edge_data>")

    if not ir_text or not edge_text:
        return Response("ERROR: malformed data\n", status=400, mimetype="text/plain")

    graph = build_graph(
        ir_text,
        edge_text,
        tokenizer,
        encoder,
    )

    edge_data = parse_score_file(edge_text)

    if graph is None:
        return Response("", status=500, mimetype="text/plain")

    with torch.no_grad():
        pred = gnn(graph.to(DEVICE)).cpu()
        print("INFER STD:", pred.std().item(), pred.min().item(), pred.max().item())
        pred = (-pred).tolist()

    # reconstruct edge names
    nodes = parse_ir_file(ir_text)
    names = sorted(nodes.keys())
    idx_to_name = {i: n for i, n in enumerate(names)}

    lines = []
    for (src_i, dst_i), score in zip(
        graph.edge_index.T.tolist(),
        pred,
    ):
        src = idx_to_name[src_i]
        dst = idx_to_name[dst_i]

        if (src, dst) not in edge_data and (dst, src) not in edge_data:
            continue

        lines.append(f"{src}, {dst}: {float(score):.6f}")

    return Response("\n".join(lines) + "\n", status=200, mimetype="text/plain")


if __name__ == "__main__":
    app.run(port=3031)
