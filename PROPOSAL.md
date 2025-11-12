### CS521: Project Proposal
### Sameer Narendran (sameern3), Alan Luo (alanluo3)

TorchInductor, the default compiler backend for PyTorch employs an operation fusion strategy that is primarily guided by a set of hand-written/hard-coded heuristics. Examples include `should_prefer_unfused_addmm`, which decides whether to fuse an `add` or `mm` operation based on subsequent operations (pointwise or not). Notably, the `score_fusion` function assigns a score to the fusion of two operators/nodes based on node proximity and memory. While effective in many cases, this heuristic-based system inherently limits the compiler’s ability to adapt to diverse model architectures and hardware targets. Therefore, we propose to replace TorchInductor’s heuristic based fusion logic with a data-driven, learned cost model.

For the model architecture, preliminary experiments will focus on tree-based models like XGBoost and LightGBM, leveraging tree architectures’ natural alignment with the compiler’s conditional logic. Our model will predict the estimated speedup of fusing two nodes given the subgraph topology, compared to the unfused variant and be trained on the MSE loss. We plan on training our models on a combination of synthetic randomly-generated graphs and subgraphs extracted from open-source Hugging Face models using `torch.fx`.

Our planned timeline for the project is as follows:
1. Exploring Pytorch codebase: 2 weeks
2. Generating sample graphs for training: 3-5 days
3. Compiling graphs from Hugging Face models for training: 5-7 days
4. Train models on graph fusion operations: 1-2 weeks
5. Integrate model into Pytorch: 2 days
6. Benchmark performance on open-source models and compile results: 1 week
