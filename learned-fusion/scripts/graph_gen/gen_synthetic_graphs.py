import json
import os
import random

import torch
import torch.fx as fx

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/graphs")
os.makedirs(DATA_DIR, exist_ok=True)


def random_op(x, y):
    # Randomly choose a binary op
    ops = [
        torch.add,
        torch.mul,
        torch.sub,
        torch.div,
        torch.matmul,
        torch.maximum,
        torch.minimum,
    ]
    return random.choice(ops)(x, y)


def generate_synthetic_module(num_nodes=5, input_shape=(4, 4)):
    """
    Generates a synthetic nn.Module with a random computation graph.
    """

    class SyntheticModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            tensors = [x]
            for i in range(num_nodes):
                # Randomly select two tensors to operate on
                t1 = random.choice(tensors)
                t2 = random.choice(tensors)
                # Apply a random operation
                out = random_op(t1, t2)
                tensors.append(out)
            # Output the last tensor
            return tensors[-1]

    return SyntheticModule(), input_shape


def serialize_fx_graph(graph_module, file_path):
    """
    Serializes an fx.GraphModule to a JSON file.
    """
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
        module, input_shape = generate_synthetic_module(
            num_nodes=random.randint(4, 8),
            input_shape=(random.randint(2, 8), random.randint(2, 8)),
        )
        traced = fx.symbolic_trace(module)
        file_path = os.path.join(DATA_DIR, f"synthetic_graph_{i}.json")
        serialize_fx_graph(traced, file_path)
        print(f"Saved synthetic graph {i} to {file_path}")


if __name__ == "__main__":
    main()
