import os
import subprocess
import glob
from typing import Optional
import argparse
import time
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch._inductor.config as iconfig


# ==================================================================
#  Configure Inductor Debug / Kernel Dump Locations / Logging Levels
# ==================================================================
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_LOGS"] = "+inductor,graph,graph_code,aot_graphs,output_code"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_SAVE_OPERATORS"] = "1"
os.environ["TRITON_SAVE_TTIR"] = "1"
os.environ["TORCHINDUCTOR_FX_COMPILE_MODE"] = "SERIALIZE"

iconfig.trace.enabled = True
iconfig.trace.graph_diagram = True

# ==============================================
# Check that all the expected output files exist
# ==============================================
def ensure_expected_files(model_path, directory) -> (bool, Optional[str]):
    expected_output_files = set([
        f"{model_path}_graph.txt",
        "fx_graph_readable.py",
        "fx_graph_runnable.py",
        "fx_graph_transformed.py",
        "ir_post_fusion.txt",
        "ir_pre_fusion.txt",
        "output_code.py",
    ])

    for root, _, files in os.walk(directory):
        expected_output_files = expected_output_files.difference(files)
    
    return len(expected_output_files) == 0, f"Missing Files: {expected_output_files}" if len(expected_output_files) != 0 else None
    

# ==========================================================
# Find all triton kernel files and call objdump if necessary
# ==========================================================
def find_triton_kernels(kernel_dir) -> (set, Optional[str]):
    kernels = set()
    kernels_to_disassemble = set()
    for root, _, files in os.walk(kernel_dir):
        for f in files:
            if f.endswith((".ttir", ".ll", ".cubin", ".asm", ".s", ".ptx")):
                kernels.add(f)
            if f.endswith((".so", ".o")):
                kernels_to_disassemble.add(os.path.join(root, f))

    for f in kernels_to_disassemble:
        f_without_ext = os.path.splitext(os.path.basename(f))[0]

        if f_without_ext in kernels:
            continue

        output_file = os.path.join(os.path.dirname(f), f"{f_without_ext}.s")

        try:
            with open(output_file, "w") as f_out:
                subprocess.run(
                    ["objdump", "-d", f],
                    stdout=f_out,
                    stderr=subprocess.PIPE,
                    check=True
                )
        except subprocess.CalledProcessError as e:
            return set(), f"Error disassembling {f}: {e.stderr.decode()}"
    
    return kernels, None


def move_inductor_logs(directory) -> (bool, Optional[str]):

    dest_dir = os.path.join(directory)
    os.makedirs(dest_dir, exist_ok=True)

    run_dirs = glob.glob("torch_compile_debug/run*")
    if not run_dirs:
        return False, "No run directories found"

    latest_run_dir = sorted(run_dirs)[-1]
    print(["mv", os.path.join(latest_run_dir, "torchinductor", "*", "*"), dest_dir])

    try:
        cmd = f"mv {os.path.join(latest_run_dir, 'torchinductor', '*', '*')} {dest_dir}"
        subprocess.run(
            cmd,
            shell=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        return False, f"Error moving files from {latest_run_dir}: {e}"

    return True, None


# ===================================
# Export Torch.FX Graph For The Model
# ===================================
def export_torchfx_graph(model, example_inputs, directory, output_file):
    print(f"\t- Exporting torch.fx graph")
    exported = torch.export.export(model, example_inputs)
    with open(os.path.join(directory, f"{output_file}"), "w") as f:
        f.write(str(exported.graph))
    

# ===========================================================
# Compile a model with torch inductor and dump graphs/kernels
# ===========================================================
def compile_model(model, example_inputs, name):

    # Compile with torch inductor and run once to make sure kernels generate
    print(f"\t- Compiling with torch inductor")
    compiled = torch.compile(model, backend="inductor", fullgraph=False)
    compiled(*example_inputs)


def is_encoder_only(model_id: str) -> bool:
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception:
        return False

    # Encoder-only check
    # https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertModel
    if getattr(config, "is_encoder_decoder", False):
        return False
    if getattr(config, "architectures", None):
        archs = [a.lower() for a in config.architectures]
        if any("gpt" in a or "llama" in a or "bloom" in a for a in archs):
            return False
    if getattr(config, "model_type", None) in ["clip", "vision", "image", "audio", "speech"]:
        return False

    return True


def get_models(dataset_path, max_samples):

    df = pd.read_csv(dataset_path).head(n=20)
    df = df[df["model_id"].apply(is_encoder_only)]

    max_samples = max_samples or len(df)
    return df["model_id"].tolist()[:max_samples]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/torch_compile_out/"
    )
    parser.add_argument(
        "--triton_kernels",
        action="store_true",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
    )
    parser.add_argument(
        "--max_samples",
        type=int
    )
    args = parser.parse_args()
    
    if args.dataset_path is None:
        print("No dataset given")
        exit(1)
    
    OUTPUT_DIR = args.output_path if args.output_path else "data/torch_compile_out/"
    TRITON_KERNEL_DIR = "triton_kernels"
    TORCH_INDUCTOR_CACHE_DIR = '/home/sameern3/torch-graph-gen/_torch_cache'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TORCH_INDUCTOR_CACHE_DIR, exist_ok=True)
    
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = TORCH_INDUCTOR_CACHE_DIR

    model_list = get_models(args.dataset_path, args.max_samples)

    output_df = pd.DataFrame(
        columns=["model", "triton_kernels", "error"]
    )


    for idx, model_name in enumerate(model_list):

        print(f"Processing {model_name} ({idx+1}/{len(model_list)})")

        model_path = model_name.replace("/", "_")
        triton_model_kernel_dir = os.path.join(OUTPUT_DIR, model_path, TRITON_KERNEL_DIR)
        ti_model_log_dir = os.path.join(OUTPUT_DIR, model_path, 'logs')

        os.makedirs(os.path.join(OUTPUT_DIR, model_path), exist_ok=True)

        os.environ["TRITON_CACHE_DIR"] = str(triton_model_kernel_dir)
        os.environ["TORCH_INDUCTOR_LOG_DIR"]= str(ti_model_log_dir)

        # print(f"=== Processing: {model_name} ===")

        t_start = time.time()

        try:
            # Load model from Huggingface
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).eval()

            # Tokenize sample inputs
            encoded = tokenizer("This is a test input.", return_tensors="pt")
            example_inputs = (encoded["input_ids"], encoded["attention_mask"])

            from_cache = True
            moved, err1 = False, None
            if args.no_cache:
                from_cache = False
                export_torchfx_graph(model, example_inputs, os.path.join(OUTPUT_DIR, model_path), f"{model_path}_graph.txt")
                compile_model(model, example_inputs, model_path)
                moved, err1 = move_inductor_logs(os.path.join(OUTPUT_DIR, model_path))
            elif not ensure_expected_files(model_path, os.path.join(OUTPUT_DIR, model_path))[0]:
                from_cache = False
                export_torchfx_graph(model, example_inputs, os.path.join(OUTPUT_DIR, model_path), f"{model_path}_graph.txt")
                compile_model(model, example_inputs, model_path)
                moved, err1 = move_inductor_logs(os.path.join(OUTPUT_DIR, model_path))
            else:
                print(f"\t- Found model in cache")

            kernels_found, err2 = find_triton_kernels(triton_model_kernel_dir)
            files_found, err3 = ensure_expected_files(model_path, os.path.join(OUTPUT_DIR, model_path))
            errs = list(filter(lambda x: x, [err1, err2, err3]))

            output_df.loc[len(output_df)] = {
                "model": model_name,
                "triton_kernels": len(kernels_found),
                "cached": from_cache,
                "error": None if len(errs) == 0 else '\n'.join(errs),
            }
        except Exception as e:
            output_df.loc[len(output_df)] = {
                "model": model_name,
                "triton_kernels": 0,
                "cached": False,
                "error": e,
            }

        print(f"\t- Finished processing {model_name} ({round(time.time() - t_start, 3)}s)")

    failed_models = output_df[output_df["error"].notna()]
    print(f"Successfully finished {len(output_df) - len(failed_models)} models.")
    print(f"Failed: {failed_models['model'].tolist()}")

    output_df.to_csv(os.path.join("data", f"generation_results_{int(time.time())}.csv"), index=False)



if __name__ == "__main__":
    main()
