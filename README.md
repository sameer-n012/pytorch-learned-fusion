# Learned Operator Fusion in TorchInductor
### UIUC FA2025: CS 521 Final Project
### Authors: Alan Luo & Sameer Narendran

Original `README.md` file moved to [PYTORCH_README.md](./PYTORCH_README.md)

The project proposal can be found at [PROPOSAL.md](./PROPOSAL.md)

The project report can be found at [report.pdf](./report.pdf)

### Project Code
The main learned fusion code is in the `learned-fusion/` directory.
- `learned-fusion/scripts/graph_gen`: Contains code for generating graphs.
- `learned-fusion/data/torch_compile_out`: Contains generated graph data.
- `learned-fusion/scripts/train`: Contains code for training the learned fusion model.
- `learned-fusion/scripts/eval`: Contains code for evaluating the learned fusion model.
- `learned-fusion/scripts/train`: Contains code for training the learned fusion model.
- `learned-fusion/output/`: Contains the training and evaluation results as well as plots

There were also edits made to the following PyTorch files to integrate the learned fusion model:
- `torch/_inductor/scheduler.py`: Modified the `get_possible_fusions` function and added the `benchmark_kernel_get_times` function.
- `torch/_inductor/choices.py`: Added the `learned_score_fusion` function.


### Building Instructions

IMPORTANT: All building and testing was performed on an AWS g5.2xlarge instance using the
Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6.0 (Amazon Linux 2023) 20250525 (AMI ID: ami-06ba49edd0fea1502)

1. Clone the repository and navigate into the pytorch-learned-fusion directory
2. Uninstall any existing installations of PyTorch by running the command `pip uninstall torch` until you see `WARNING: Skipping torch as it is not installed`.
3. Install miniconda if it is not already installed. See [installation instructions](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2)
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
You can either use the base conda environment or create a new one.

4. Run `python setup.py clean`
5. Run `git submodule sync` and then `git submodule update --init --recursive`
6. Install dependencies by running `pip install --group dev`
7. Run `pip install mkl-static mkl-include`
8. Run `.ci/docker/common/install_magma_conda.sh 12.4` if using CUDA
9. Run `pip install triton`. You may need to run `make triton` afterwards.
10. Run `pip install ninja`.
11. Run `pip install --no-build-isolation -v -e .`. This will take a while the first time.
12. Navigate to the `learned-fusion` directory and install any additional dependencies you need to run your scripts.
The learned fusions scripts require you to run `pip install -r requirements.txt` in the `learned-fusion` directory.

See [Contributing.md](./CONTRIBUTING.md) for in-depth build details or troubleshooting.


### Running the Learned Fusion Model
1. After building PyTorch as described above, navigate to the `learned-fusion/` directory.
2. Run `./scripts/training/serve_gnn.py` to start the learned fusion server.
3. In a separate terminal, run `./scripts/example.py` to compile a model using the learned fusion server.
Note that if you want to compile the same model multiple times, you will need to clear the TorchInductor cache.
This can be done by setting the environment variable `TORCHINDUCTOR_CACHE_DIR` to a new directory before compiling anything,
and deleting the directory afterwards.
You can use the default TorchInductor fusion heuristics by setting the environment variable `TORCH_USE_DEFAULT_SCORE_FUSION` to `1`.
You can use a random fusion strategy by setting the environment variable `TORCH_USE_RANDOM_SCORE_FUSION` to `1`.
Setting the default score fusion variable will override the random score fusion variable if both are set.
