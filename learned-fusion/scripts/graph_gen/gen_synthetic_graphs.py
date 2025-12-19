import argparse
import glob
import json
import os
import random
import re
import subprocess
import time
import traceback
from typing import Optional

import pandas as pd
import torch
import torch._inductor.config as iconfig
import torch.fx as fx

DATA_DIR = "./data/synthetic/"
os.makedirs(DATA_DIR, exist_ok=True)


os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_LOGS"] = "+inductor,graph,graph_code,aot_graphs,output_code"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_SAVE_OPERATORS"] = "1"
os.environ["TRITON_SAVE_TTIR"] = "1"
os.environ["TORCHINDUCTOR_FX_COMPILE_MODE"] = "SERIALIZE"

os.environ["TRITON_CACHE_DIR"] = os.path.join(DATA_DIR, "triton_kernels")
os.environ["TORCH_INDUCTOR_LOG_DIR"] = os.path.join(DATA_DIR, "logs")
os.environ["MY_TORCH_MODEL_OUTPUT_DIR"] = str(
    os.path.join(DATA_DIR, "torch_compile_out")
)

os.makedirs(os.path.join(DATA_DIR, "torch_compile_out"), exist_ok=True)

iconfig.trace.enabled = True
iconfig.trace.graph_diagram = True


def random_unop():
    # Randomly choose a unary op
    ops = [
        # Math
        torch.abs,
        torch.neg,
        torch.sin,
        torch.cos,
        torch.tan,
        torch.exp,
        torch.log,
        torch.sqrt,
        torch.sigmoid,
        torch.tanh,
        torch.relu,
        torch.floor,
        torch.ceil,
        torch.round,
        torch.frac,
        # Tensor manipulations
        lambda x: x.transpose(0, 1),
        # lambda x: x.permute(*reversed(range(x.dim()))),
        # lambda x: x.reshape(-1, x.shape[-1]),
        # lambda x: x.unsqueeze(0),
        # lambda x: x.squeeze(),
        lambda x: torch.clamp(x, min=0, max=10),
        lambda x: torch.softmax(x, dim=-1),
        lambda x: torch.log_softmax(x, dim=-1),
    ]

    ops = [lambda x, y: op(x) for op in ops]

    return random.choice(ops)


def random_binop():
    # Randomly choose a binary op
    ops = [
        # Arithmetic
        torch.add,
        torch.sub,
        torch.mul,
        torch.div,
        # torch.remainder,
        # torch.fmod,
        torch.pow,
        # Max / Min
        torch.maximum,
        torch.minimum,
        # Linear algebra
        # lambda a, b: torch.matmul(*make_matmul_compat(a, b)),
        # lambda a, b: torch.mm(*make_mm_compat(a, b)),
        # lambda a, b: torch.bmm(*make_bmm_compat(a, b)),
    ]
    return random.choice(ops)


def make_matmul_compat(a, b):
    # Ensure 2D tensors for mm, matmul works with 1D and higher
    a_dim, b_dim = a.dim(), b.dim()

    if a_dim == 1:
        a = a.unsqueeze(0)  # make it 2D
    if b_dim == 1:
        b = b.unsqueeze(1)  # make it 2D

    # Match inner dimensions by slicing if needed
    if a.shape[-1] != b.shape[-2]:
        min_dim = min(a.shape[-1], b.shape[-2])
        a = a[..., :min_dim]
        b = b[..., :min_dim, :] if b.dim() > 1 else b

    return a, b


def make_mm_compat(a, b):
    # ensure 2D
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])
    min_dim = min(a.shape[1], b.shape[0])
    a = a[:, :min_dim]
    b = b[:min_dim, :]
    return a, b


def generate_synthetic_module(num_nodes=5, input_shape=(4, 4)):
    op_list = [
        random_unop() if random.random() < 0.5 else random_binop()
        for _ in range(num_nodes)
    ]

    class SyntheticModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            tensors = [x]
            for i in range(num_nodes):
                t1 = random.choice(tensors)
                t2 = random.choice(tensors)
                op = random_binop()
                print(op)
                tensors.append(op(t1, t2))

            return tensors[-1]

    return SyntheticModule(), input_shape


def generate_sample_input(input_shape):
    return torch.randn(*input_shape)


def compile_synthetic(model, example_input):
    model = model.to("cpu")

    compiled = torch.compile(model, backend="inductor", fullgraph=False)
    return compiled(example_input)


def serialize_fx_graph(graph_module, file_path):
    graph_dict = {
        "nodes": [],
        "name": graph_module.__class__.__name__,
    }
    for node in graph_module.graph.nodes:
        node_dict = {
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "args": str(node.args),
            "kwargs": str(node.kwargs),
        }
        graph_dict["nodes"].append(node_dict)
    with open(file_path, "w") as f:
        json.dump(graph_dict, f, indent=2)


def main():
    num_graphs = 10
    for i in range(num_graphs):
        try:
            module, input_shape = generate_synthetic_module(
                num_nodes=random.randint(4, 8),
                input_shape=(random.randint(2, 8), random.randint(2, 8)),
                num_nodes=100,
                input_shape=(64, 64),
            )

            example_input = generate_sample_input(input_shape).to("cuda")
            module = module.to("cuda")
            x = compile_synthetic(module, example_input)
            print(x)

        except Exception as e:
            print(f"Error generating or compiling synthetic graph {i}: {e}")
            traceback.print_exc()
            # exit()
            continue

        # traced = fx.symbolic_trace(module)
        # file_path = os.path.join(DATA_DIR, f"synthetic_graph_{i}.json")
        # serialize_fx_graph(traced, file_path)
        # print(f"Saved synthetic graph {i} to {file_path}")


if __name__ == "__main__":
    main()
