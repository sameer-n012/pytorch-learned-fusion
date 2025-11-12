# Learned Operator Fusion in TorchInductor
### UIUC FA2025: CS 521 Final Project
### Authors: Alan Luo & Sameer Narendran

Original `README.md` file moved to [PYTORCH_README.md](./PYTORCH_README.md)

The project proposal can be found at [PROPOSAL.md](./PROPOSAL.md)

### Project Structure
- Relevant code for learned fusion:
    - `torch/_inductor/autoheuristic/`
    - `torch/_inductor/choices.py`
    - `torch/_inductor/scheduler.py`
    - `torch/_inductor/graph.py`
    - `torch/_inductor/fx_passes/`
    - `torch/fx/graph.py`
- Relevant code for testing:
    - `test/inductor/`
    - `test/fx/`
- `learned-fusion/models`: Contains the models used for learned fusion.
- `learned-fusion/graph_gen`: Contains code for generating graphs.
- `learned-fusion/data/graphs`: Contains generated graph data.
- `learned-fusion/data/results`: Contains results from training and evaluation.
- `learned-fusion/scripts`: Contains code for training/evaluating the learned fusion model.


### Building from Source
1. Clone the repository
2. Uninstall any existing installations of PyTorch by running the command `pip uninstall torch` until you see `WARNING: Skipping torch as it is not installed`.
3. Run `python setup.py clean`
4. Run `git submodule sync` and then `git submodule update --init --recursive`

For Linux:
1. Install dependencies by running `pip install --group dev`
2. Run `pip install mkl-static mkl-include`,
3. Run `.ci/docker/common/install_magma_conda.sh 12.4` if using CUDA
4. Install the appropriate version of triton and run `make triton` from the pytorch repository root.
5. Build with `python -m pip install -e . -v --no-build-isolation`
6. Run `export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"`
7. Run `python -m pip install --no-build-isolation -v -e .`

For MacOS:
1. Install dependencies by running `pip install --group dev`
2. Run `pip install mkl-static mkl-include` if using an Intel x86 Mac.
3. Run `conda install pkg-config libuv` if torch.distributed is to be used.
4. Run `python -m pip install --no-build-isolation -v -e .`

For incremental building, you can use `ninja`, `ccache`, and `mold`/`lld` to speed up the build process.

Ninja:
1. Install ninja with `pip install ninja`
2. You may need to run `python setup.py clean` once before using ninja for the first time.

CCache:
1. Install ccache with `apt`, `yum`, or `brew`
2. You may need to configure the cache size:
```sh
# config: cache dir is ~/.ccache, conf file ~/.ccache/ccache.conf
# max size of cache
ccache -M 25Gi  # -M 0 for unlimited
# unlimited number of files
ccache -F 0
```
3. Check that the following variables contain `ccache`: `CMAKE_C_COMPILER_LAUNCHER`, `CMAKE_CXX_COMPILER_LAUNCHER`, and `CMAKE_CUDA_COMPILER_LAUNCHER`. If not, set them to `ccache`.

Mold:
1. Install mold with `apt` or `yum`. Note that it requires GCC version of 12 or higher.
2. Set the variable `CMAKE_LINKER_TYPE=MOLD`

LLD:
1. Install LLD from the [LLVM Release](https://releases.llvm.org/download.html).
2. Set the variable `CMAKE_LINKER_TYPE=LLD`

See [Contributing.md](./CONTRIBUTING.md) for in-depth build details or troubleshooting.

### Testing
To run tests, make sure `expecttest` and `hypothesis` are installed with pip. These should have been installed with `pip install --group dev`
- You can run all tests with `python test/run_test.py`
- You can run a specific test file with `python test/FILENAME.py` (e.g. `python test/test_jit.py`)
- You can run a specific test class with `python test/FILENAME.py TESTCLASSNAME.TESTNAME` (e.g. `python test/test_jit.py TestJit.test_Sequential`)

See [Contributing.md](./CONTRIBUTING.md) for in-depth build details or troubleshooting.
