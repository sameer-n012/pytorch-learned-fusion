import argparse
import glob
import os
import re
import subprocess
import time
import traceback
from typing import Optional

import pandas as pd
import torch
import torch._inductor.config as iconfig
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GPTNeoXForCausalLM, GPTNeoXTokenizerFast, AutoModelForSequenceClassification, AutoModelForMaskedLM

# ==================================================================
#  Configure Inductor Debug / Kernel Dump Locations / Logging Levels
# ==================================================================
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_LOGS"] = "+inductor,graph,graph_code,aot_graphs,output_code"
# os.environ["TORCH_LOGS"] = "+inductor,graph"
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
    expected_output_files = set(
        [
            # f"{model_path}_graph.txt",
            # "fx_graph_readable.py",
            # "fx_graph_runnable.py",
            # "fx_graph_transformed.py",
            # "ir_post_fusion.txt",
            # "ir_pre_fusion.txt",
            # "output_code.py",
        ]
    )

    found_my_score_fusion = False
    found_my_ir_fusion = False

    for root, _, files in os.walk(directory):
        expected_output_files = expected_output_files.difference(files)
        for f in files:
            if f.startswith("my_score_fusions_test_file") and f.endswith(".txt"):
                found_my_score_fusion = True
            if f.startswith("my_ir_test_file") and f.endswith(".txt"):
                found_my_ir_fusion = True

    found_all = (
        len(expected_output_files) == 0 and found_my_score_fusion and found_my_ir_fusion
    )
    missing_str = ""
    if not found_my_score_fusion:
        missing_str += " Missing my_score_fusions_test_file."
    if not found_my_ir_fusion:
        missing_str += " Missing my_ir_test_file."
    if len(expected_output_files) > 0:
        missing_str += f" Missing files: {expected_output_files}."

    # if model_path == "roberta-large-mnli":
    #     print(f"Debug: found_my_score_fusion={found_my_score_fusion}, found_my_ir_fusion={found_my_ir_fusion}, expected_output_files={expected_output_files}")
    #     print(' '.join(os.listdir(directory)))
    # exit()

    return (found_all, missing_str if not found_all else None)


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
                    check=True,
                )
        except subprocess.CalledProcessError as e:
            return set(), f"Error disassembling {f}: {e.stderr.decode()}"

    return kernels, None


def try_remove_model_dir(output_dir, model_path):
    model_dir = os.path.join(output_dir, model_path)
    try:
        if os.path.exists(model_dir):
            subprocess.run(["rmdir", model_dir], check=True)
    except Exception as e:
        return


def get_model_size(model_name: str, dtype: str = "float32") -> float:
    cmd = ["accelerate", "estimate-memory", model_name, "--dtypes", dtype]

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )

        output = result.stdout + "\n" + result.stderr

        # example row:
        # │float32│   86.47 MB  │413.18 MB │      1.61 GB      │
        pattern = r"\│\s*{dtype}\s*\│.*?\│\s*([0-9.]+)\s*(KB|MB|GB)\s*\│".format(
            dtype=re.escape(dtype)
        )

        match = re.search(pattern, output)
        if not match:
            return None

        size_value = float(match.group(1))
        size_unit = match.group(2).upper()

        if size_unit == "KB":
            size_mb = size_value / 1024
        elif size_unit == "MB":
            size_mb = size_value
        elif size_unit == "GB":
            size_mb = size_value * 1024
        else:
            return None

        return size_mb

    except subprocess.CalledProcessError as e:
        return None


def move_inductor_logs(directory) -> (bool, Optional[str]):
    dest_dir = os.path.join(directory)
    os.makedirs(dest_dir, exist_ok=True)

    run_dirs = glob.glob("torch_compile_debug/run*")
    if not run_dirs:
        return False, "No run directories found"

    latest_run_dir = sorted(run_dirs)[-1]

    try:
        cmd = f"mv {os.path.join(latest_run_dir, 'torchinductor', '*', '*')} {dest_dir}"
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return False, f"Error moving files from {latest_run_dir}: {e}"

    return True, None


# ===================================
# Export Torch.FX Graph For The Model
# ===================================
def export_torchfx_graph(
    model, example_inputs, example_inputs_dict, directory, output_file
):
    print(f"\t- Exporting torch.fx graph")
    try:
        exported = torch.export.export(model, example_inputs)
        with open(os.path.join(directory, f"{output_file}"), "w") as f:
            f.write(str(exported.graph))
    except TypeError as e:
        if "unexpected positional argument" in str(e):
            exported = torch.export.export(model, args=(), kwargs=example_inputs_dict)
            with open(os.path.join(directory, f"{output_file}"), "w") as f:
                f.write(str(exported.graph))
        else:
            raise


# ===========================================================
# Compile a model with torch inductor and dump graphs/kernels
# ===========================================================
def compile_model(model, example_inputs_dict, name):
    # Compile with torch inductor and run once to make sure kernels generate
    print(f"\t- Compiling with torch inductor")
    compiled = torch.compile(model, backend="inductor", fullgraph=False)
    compiled(**example_inputs_dict)


def is_encoder_only(model_id: str) -> bool:
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception:
        return False

    # Encoder-only check
    # https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertModel
    # if getattr(config, "is_encoder_decoder", False):
    #     return False
    # if getattr(config, "architectures", None):
    #     archs = [a.lower() for a in config.architectures]
    #     if any("gpt" in a or "llama" in a or "bloom" in a for a in archs):
    #         return False
    if getattr(config, "model_type", None) in [
        "clip",
        "vision",
        "image",
        "audio",
        "speech",
    ]:
        return False

    return True


def get_classes_by_architecture(config):
    if config.is_encoder_decoder:
        return AutoTokenizer, AutoModelForSeq2SeqLM

    model_type = config.model_type

    if model_type in {"bert", "roberta", "deberta", "deberta-v2", "electra",
                      "albert", "mpnet", "minilm", "ernie", "biobert"}:
        return AutoTokenizer, AutoModel
    elif model_type in {"gpt2", "gpt_neo", "gptj", "bloom", "opt", "llama", "phi", "mt5", "t5"}:
        return AutoTokenizer, AutoModelForCausalLM
    elif model_type in {"gpt_neox"}:
        return GPTNeoXTokenizerFast, GPTNeoXForCausalLM
    elif model_type in {"mra"}:
        return AutoTokenizer, AutoModelForMaskedLM

    # Unknown → assume encoder
    return AutoTokenizer, AutoModel


def get_models_as_chunks(dataset_path, chunk_size):
    iter = pd.read_csv(dataset_path, chunksize=chunk_size)
    return iter


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
    parser.add_argument("--output_path", type=str, default="data/torch_compile_out/")
    parser.add_argument(
        "--triton_kernels",
        action="store_true",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
    )
    parser.add_argument("--max_samples", type=int)
    args = parser.parse_args()

    run_id = int(time.time())

    if args.dataset_path is None:
        print("No dataset given")
        exit(1)

    OUTPUT_DIR = args.output_path if args.output_path else "data/torch_compile_out/"
    TRITON_KERNEL_DIR = "triton_kernels"
    TORCH_INDUCTOR_CACHE_DIR = (
        "/home/ec2-user/final/pytorch-learned-fusion/learned-fusion/_torch_cache"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TORCH_INDUCTOR_CACHE_DIR, exist_ok=True)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = TORCH_INDUCTOR_CACHE_DIR

    # model_list = get_models(args.dataset_path, args.max_samples)
    # model_list = ["bert-base-uncased"]

    output_df = pd.DataFrame(columns=["model", "triton_kernels", "cached", "error"])

    # for chunk in get_models_as_chunks(args.dataset_path, chunk_size=1000):
    for chunk in range(1):
        # model_list = []
        # for _, row in chunk.iterrows():
        #     model_list.append(row["model_id"])
        model_list = [
            "bert-base-uncased",
            "bert-large-uncased",
            "bert-base-cased",
            "bert-large-cased",
            "bert-base-multilingual-cased",
            "bert-base-multilingual-uncased",
            "google-bert/bert-base-uncased",
            "google-bert/bert-base-cased",

            # DistilBERT
            "distilbert-base-uncased",
            "distilbert-base-cased",
            "distilbert-base-multilingual-cased",
            "distilbert-base-german-cased",

            # RoBERTa
            "roberta-base",
            "roberta-large",
            "roberta-large-mnli",
            "roberta-base-openai-detector",
            "cardiffnlp/twitter-roberta-base",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",

            # ALBERT
            "albert-base-v1",
            "albert-large-v1",
            "albert-xlarge-v1",
            "albert-base-v2",
            "albert-large-v2",
            "albert-xlarge-v2",

            # DeBERTa
            "microsoft/deberta-base",
            "microsoft/deberta-large",
            "microsoft/deberta-v3-base",
            "microsoft/deberta-v3-large",
            "microsoft/deberta-v2-xlarge",
            # "microsoft/deberta-v2-xxlarge", # TODO FAILED too large

            # Electra
            "google/electra-small-discriminator",
            "google/electra-base-discriminator",
            "google/electra-large-discriminator",
            "google/electra-small-generator",
            "google/electra-base-generator",

            # GPT-2
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "distilgpt2",

            # GPT-Neo / GPT-J small variants
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
            # "EleutherAI/gpt-j-6B", # TODO FAILED too large

            # T5
            # REQUIRES USING AutoModelForSeq2SeqLM
            "t5-small",
            "t5-base",
            "t5-large",
            "google/flan-t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
            "google/flan-t5-xl",
            # "google/flan-t5-xxl",  # TODO FAILED too large

            # Longformer
            "allenai/longformer-base-4096",
            "allenai/longformer-large-4096",

            # XLNet
            "xlnet-base-cased",
            "xlnet-large-cased",

            # MiniLM
            "nreimers/MiniLM-L6-H384-uncased",
            "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base",
            "nreimers/MiniLMv2-L12-H384-distilled-from-RoBERTa-Large",
            "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large",

            # MobileBERT
            "google/mobilebert-uncased",

            # Funnel Transformer
            "funnel-transformer/small",
            "funnel-transformer/medium",
            "funnel-transformer/intermediate",
            "funnel-transformer/large",

            # CamemBERT (French RoBERTa)
            "camembert-base",
            "camembert/camembert-large",

            # XLM / XLM-R
            "xlm-mlm-17-1280",
            "xlm-mlm-100-1280",
            "xlm-roberta-base",
            "xlm-roberta-large",

            # ERNIE
            "nghuyong/ernie-1.0-base-zh",
            "nghuyong/ernie-2.0-base-en",

            # BART
            "facebook/bart-base",
            "facebook/bart-large",
            "facebook/bart-large-mnli",
            "facebook/bart-large-cnn",

            # MarianMT
            # REQUIRES ONLY PASSING input_ids to compiled()
            "Helsinki-NLP/opus-mt-en-de",
            "Helsinki-NLP/opus-mt-de-en",
            "Helsinki-NLP/opus-mt-en-fr",
            "Helsinki-NLP/opus-mt-en-es",
            "Helsinki-NLP/opus-mt-mul-en",

            # Bloom small models
            "bigscience/bloom-560m",
            "bigscience/bloom-1b1",
            # "bigscience/bloom-1b7", # TODO FAILED too large
            # "bigscience/bloom-3b", # TODO FAILED too large

            # # GPT2-alike
            "facebook/opt-125m",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            # # "facebook/opt-6.7b", TODO FAILED too large

            # Phi small models
            "microsoft/phi-1_5",
            "microsoft/phi-2",
            "microsoft/phi-3-mini-4k-instruct",
            # "microsoft/phi-3-small-8k-instruct", # TODO FAILED too large

            # Mistral small TODO FAILED too large
            # "mistralai/Mistral-7B-v0.1",
            # "mistralai/Mistral-7B-v0.2",
            # "mistralai/Mixtral-8x7B-v0.1",

            # ESM
            "facebook/esm2_t6_8M_UR50D",
            "facebook/esm2_t12_35M_UR50D",
            "facebook/esm2_t33_650M_UR50D",

            # Code Models
            "microsoft/codebert-base",
            "neulab/codebert-c",
            "huggingface/CodeBERTa-small-v1",

            # Sentence Transformers
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/all-distilroberta-v1",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",

            # More BERT Variants
            "bert-base-chinese",
            "tohoku-nlp/bert-base-japanese-char-v3",
            "cl-tohoku/bert-base-japanese-v3",
            "tohoku-nlp/bert-large-japanese-v2",
            "bert-base-german-cased",
            "dbmdz/bert-base-turkish-cased",
            "GroNLP/bert-base-dutch-cased",

            # More DistilBERT Variants
            "dccuchile/distilbert-base-spanish-uncased",
            "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            "huggingface/distilbert-base-uncased-finetuned-mnli",

            # Chinese BERT-like Models
            "hfl/chinese-bert-wwm-ext",
            "hfl/chinese-roberta-wwm-ext",
            "hfl/chinese-macbert-base",
            "hfl/chinese-macbert-large",

            # More RoBERTa Variants
            "roberta-base-openai-detector",
            "squeezebert/squeezebert-uncased",
            "squeezebert/squeezebert-mnli",
            "siebert/sentiment-roberta-large-english",

            # GPT-NeoX Small Models
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b",

            # RWKV
            # "RWKV/rwkv-430m-world",
            # "RWKV/rwkv-1b5-world",
            # "RWKV/rwkv-2b-world",
            "RWKV/RWKV7-Goose-World2.8-0.1B-HF",
            "RWKV/RWKV7-Goose-World2.9-0.4B-HF",
            "RWKV/RWKV7-Goose-World3-1.5B-HF",

            # mT5 Variants
            "google/mt5-small",
            "google/mt5-base",
            "google/mt5-large",

            # Long-sequence Transformers
            "yikuan8/Clinical-Longformer",
            "allenai/led-base-16384",
            "allenai/led-large-16384",

            # BERT-based Biomedical Models
            "dmis-lab/biobert-base-cased-v1.1",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "emilyalsentzer/Bio_ClinicalBERT",

            # More Code Models
            "Salesforce/codet5-small",
            "Salesforce/codet5-base",

            # MPNet
            "microsoft/mpnet-base",
            "sentence-transformers/all-mpnet-base-v1",

            # LLaMA Models
            # REQUIRES removal of use_fast from tokenizer
            "unsloth/Llama-3.2-1B",
            "unsloth/Llama-3.2-3B",
            # "meta-llama/Llama-Guard-2-1B", # TODO FAILED requires hf login

            # BLOOMZ
            "bigscience/bloomz-560m",
            "bigscience/bloomz-1b1",

            # MDeBERTa
            "microsoft/mdeberta-v3-base",

            # MBart More Models
            "facebook/mbart-large-cc25",
            "facebook/mbart-large-50",

            # Pegasus
            "google/pegasus-large",
            "google/pegasus-xsum",
            "google/pegasus-multi_news",

            # GLM-Style Models TODO FAILED not found
            # "THUDM/glm-130m",
            # "THUDM/glm-515m",

            # Reformer
            "google/reformer-crime-and-punishment",

            # ConvBERT
            "YituTech/conv-bert-base",
            "YituTech/conv-bert-small",

            # Nyströmformer
            "uw-madison/nystromformer-512",
            "uw-madison/nystromformer-1024",
            "uw-madison/nystromformer-2048",

            # LayoutLM
            "microsoft/layoutlm-base-uncased",
            # "microsoft/layoutlmv2-base-uncased", # TODO FAILED requires detectron2

        ]

        for idx, model_name in enumerate(model_list):
            print(f"Processing {model_name} ({idx + 1}/{len(model_list)})")

            # if not is_encoder_only(model_name):
            #     print(f"\t- Skipping non-encoder-only model")
            #     continue

            # model_size = get_model_size(model_name)
            # if model_size is None or model_size > 20000:  # 20 GB
            #     print(f"\t- Skipping model larger than 20 GB ({model_size} MB)")
            #     continue

            model_path = model_name.replace("/", "_")

            triton_model_kernel_dir = os.path.join(
                OUTPUT_DIR, model_path, TRITON_KERNEL_DIR
            )
            ti_model_log_dir = os.path.join(OUTPUT_DIR, model_path, "logs")

            os.makedirs(os.path.join(OUTPUT_DIR, model_path), exist_ok=True)

            os.environ["TRITON_CACHE_DIR"] = str(triton_model_kernel_dir)
            os.environ["TORCH_INDUCTOR_LOG_DIR"] = str(ti_model_log_dir)
            os.environ["MY_TORCH_MODEL_OUTPUT_DIR"] = str(
                os.path.join(OUTPUT_DIR, model_path)
            )

            # print(f"=== Processing: {model_name} ===")

            t_start = time.time()

            # x = ensure_expected_files(model_path, os.path.join(OUTPUT_DIR, model_path))
            # if not x[0]:
            #     print(x)

            if (
                ensure_expected_files(model_path, os.path.join(OUTPUT_DIR, model_path))[
                    0
                ]
                and not args.no_cache
            ):
                print(f"\t- Found model in cache")
                kernels_found, err1 = find_triton_kernels(triton_model_kernel_dir)
                errs = list(filter(lambda x: x, [err1]))

                output_df.loc[len(output_df)] = {
                    "model": model_name,
                    "triton_kernels": len(kernels_found),
                    "cached": True,
                    "error": None if len(errs) == 0 else "\n".join(errs),
                }

                print(
                    f"\t- [SUCCESS] Finished processing {model_name} ({round(time.time() - t_start, 3)}s)"
                )
                continue

            try:
                config = AutoConfig.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
                tokenizer_class, model_class = get_classes_by_architecture(config)

                # Load model from Huggingface
                tokenizer = tokenizer_class.from_pretrained(
                    model_name,
                    use_fast=False,
                    trust_remote_code=True,
                    # load_in_4bit=True,
                )
                model = model_class.from_pretrained(
                    model_name,
                    trust_remote_code=True
                ).eval().to("cuda")

                # Tokenize sample inputs
                encoded = tokenizer("This is a test input.", return_tensors="pt")
                encoded_dict = {k: v.to("cuda") for k, v in encoded.items()}
                example_inputs = (
                    encoded_dict["input_ids"],
                    encoded_dict["attention_mask"],
                )
                # example_inputs_dict = {k: v.to("cuda") for k, v in encoded.items()}
                example_inputs_dict = encoded_dict

                from_cache = True
                moved, err1 = False, None
                if args.no_cache:
                    from_cache = False
                    # export_torchfx_graph(
                    #     model,
                    #     example_inputs,
                    #     example_inputs_dict,
                    #     os.path.join(OUTPUT_DIR, model_path),
                    #     f"{model_path}_graph.txt",
                    # )
                    compile_model(model, example_inputs_dict, model_path)
                    time.sleep(2)  # wait for files to flush
                    moved, err1 = move_inductor_logs(
                        os.path.join(OUTPUT_DIR, model_path)
                    )
                elif not ensure_expected_files(
                    model_path, os.path.join(OUTPUT_DIR, model_path)
                )[0]:
                    from_cache = False
                    # export_torchfx_graph(
                    #     model,
                    #     example_inputs,
                    #     example_inputs_dict,
                    #     os.path.join(OUTPUT_DIR, model_path),
                    #     f"{model_path}_graph.txt",
                    # )
                    compile_model(model, example_inputs_dict, model_path)
                    time.sleep(2)  # wait for files to flush
                    moved, err1 = move_inductor_logs(
                        os.path.join(OUTPUT_DIR, model_path)
                    )
                else:
                    print(f"\t- Found model in cache")

                kernels_found, err2 = find_triton_kernels(triton_model_kernel_dir)
                files_found, err3 = ensure_expected_files(
                    model_path, os.path.join(OUTPUT_DIR, model_path)
                )
                errs = list(filter(lambda x: x, [err1, err2, err3]))

                output_df.loc[len(output_df)] = {
                    "model": model_name,
                    "triton_kernels": len(kernels_found),
                    "cached": from_cache,
                    "error": None if len(errs) == 0 else "\n".join(errs),
                }

                print(
                    f"\t- [SUCCESS] Finished processing {model_name} ({round(time.time() - t_start, 3)}s)"
                )

                # if model:
                #     try:
                #         del model
                #     except:
                #         pass
                # if tokenizer:
                #     try:
                #         del tokenizer
                #     except:
                #         pass
                torch.cuda.empty_cache()

                output_df.to_csv(
                    os.path.join("data", f"generation_results_{run_id}.csv"),
                    index=False,
                )

                if not from_cache:
                    time.sleep(2)

            except Exception as e:
                # try:
                #     del model
                # except:
                #     pass
                # try:
                #     del tokenizer
                # except:
                #     pass
                torch.cuda.empty_cache()

                output_df.loc[len(output_df)] = {
                    "model": model_name,
                    "triton_kernels": 0,
                    "cached": False,
                    "error": e,
                }

                print(
                    f"\t- [FAIL] Finished processing {model_name} ({round(time.time() - t_start, 3)}s)"
                )
                print(traceback.format_exc())

                try_remove_model_dir(OUTPUT_DIR, model_path)

                output_df.to_csv(
                    os.path.join("data", f"generation_results_{run_id}.csv"),
                    index=False,
                )

                time.sleep(2)

    failed_models = output_df[output_df["error"].notna()]
    print(f"Successfully finished {len(output_df) - len(failed_models)} models.")
    print(f"Failed: {failed_models['model'].tolist()}")

    output_df.to_csv(
        os.path.join("data", f"generation_results_{run_id}.csv"), index=False
    )


if __name__ == "__main__":
    main()
