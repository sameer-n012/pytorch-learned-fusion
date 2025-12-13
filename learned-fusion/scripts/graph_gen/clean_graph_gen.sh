#!/bin/bash

read -p "Are you sure you want to delete learned-fusion/torch_compile_debug and learned-fusion/_torch_cache? [y/N] " confirm

if [[ "$confirm" =~ ^[Yy]$ ]]; then
    rm -rf ./torch_compile_debug
    rm -rf ./_torch_cache
    # rm -rf ./data/generation_results_*.csv
    rm -rf ./my_ir_test_file_*.txt
    rm -rf ./my_score_fusions_test_file_*.txt
    echo "Files deleted."
else
    echo "Deletion cancelled."
fi
