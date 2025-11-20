#!/bin/bash

read -p "Are you sure you want to delete ~/torch-graph-gen/torch_compile_debug and ~/torch-graph-gen/_torch_cache? [y/N] " confirm

if [[ "$confirm" =~ ^[Yy]$ ]]; then
    rm -rf ~/torch-graph-gen/torch_compile_debug
    rm -rf ~/torch-graph-gen/_torch_cache
    rm -rf ~/torch-graph-gen/data/torch_compile_debug
    rm -rf ~/torch-graph-gen/data/generation_results_*.csv
    echo "Files deleted."
else
    echo "Deletion cancelled."
fi