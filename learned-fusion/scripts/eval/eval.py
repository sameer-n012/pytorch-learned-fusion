import argparse
import glob
import os
import re
import subprocess
import time
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import torch._inductor.config as iconfig
from eval_models import model_list
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPTNeoXForCausalLM,
    GPTNeoXTokenizerFast,
)

OUTPUT_DIR = "output/eval"

MAX_LEN = 64

warmup = "This is a warmup input."
inputs = [
    "I am that merry wanderer of the night",
    "I jest to Oberon and make him smile",
    "When I a fat and bean-fed horse beguile",
    "Neighing in likeness of a filly foal",
    "And sometime lurk I in a gossip's bowl",
    "In very likeness of a roasted crab",
    "And when she drinks, against her lips I bob",
    "And on her wither'd dewlap pour the ale",
    "The wisest aunt, telling the saddest tale",
    "Sometime for three-foot stool mistaketh me",
]


# ===========================================================
# Compile a model with torch inductor and dump graphs/kernels
# ===========================================================
def compile_model(model, example_inputs_dict, name):
    # Compile with torch inductor and run once to make sure kernels generate
    s = (
        "default"
        if os.environ.get("TORCH_USE_DEFAULT_SCORE_FUSION")
        else "learned fusion"
    )
    print(f"\t- Compiling with torch inductor ({s})")
    ts = time.time()
    compiled = torch.compile(model, backend="inductor", fullgraph=False)
    compiled(**example_inputs_dict)
    tc = time.time() - ts

    return compiled, tc


def time_model_run(model, tokenizer, repeats=10):
    print(f"\t- Timing model over {repeats} runs")
    run_times = []

    warmup_encoded = tokenizer(
        warmup, return_tensors="pt", padding="max_length", max_length=MAX_LEN
    )
    warmup_dict = {k: v.to("cuda") for k, v in warmup_encoded.items()}

    encoded = [
        tokenizer(i, return_tensors="pt", padding="max_length", max_length=MAX_LEN)
        for i in inputs
    ]
    encoded_dict = [{k: v.to("cuda") for k, v in e.items()} for e in encoded]

    # warmup
    with torch.no_grad():
        model(**warmup_dict)

    pbar = tqdm(range(repeats), desc="Timing runs")
    for i, _ in enumerate(pbar):
        torch.cuda.synchronize()
        t_start = time.time()

        with torch.no_grad():
            model(**encoded_dict[i])

        torch.cuda.synchronize()
        t_end = time.time()
        run_times.append(t_end - t_start)

    return run_times


def get_classes_by_architecture(config):
    if config.is_encoder_decoder:
        return AutoTokenizer, AutoModelForSeq2SeqLM

    model_type = config.model_type

    if model_type in {
        "bert",
        "roberta",
        "deberta",
        "deberta-v2",
        "electra",
        "albert",
        "mpnet",
        "minilm",
        "ernie",
        "biobert",
    }:
        return AutoTokenizer, AutoModel
    elif model_type in {
        "gpt2",
        "gpt_neo",
        "gptj",
        "bloom",
        "opt",
        "llama",
        "phi",
        "mt5",
        "t5",
    }:
        return AutoTokenizer, AutoModelForCausalLM
    elif model_type in {"gpt_neox"}:
        return GPTNeoXTokenizerFast, GPTNeoXForCausalLM
    elif model_type in {"mra"}:
        return AutoTokenizer, AutoModelForMaskedLM

    # Unknown → assume encoder
    return AutoTokenizer, AutoModel


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--default-fusion",
        action="store_true",
        help="Use default score fusion for torch inductor",
    )
    argparser.add_argument(
        "--random-fusion",
        action="store_true",
        help="Use random score fusion for torch inductor",
    )
    args = argparser.parse_args()

    if args.default_fusion:
        os.environ["TORCH_USE_DEFAULT_SCORE_FUSION"] = "1"
    elif args.random_fusion:
        os.environ["TORCH_USE_RANDOM_SCORE_FUSION"] = "1"

    TORCH_INDUCTOR_CACHE_DIR = (
        "/home/ec2-user/final/pytorch-learned-fusion/learned-fusion/_torch_cache"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TORCH_INDUCTOR_CACHE_DIR, exist_ok=True)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = TORCH_INDUCTOR_CACHE_DIR

    run_id = int(time.time())

    output_df = pd.DataFrame(
        columns=[
            "model",
            "score_fusion_type",
            "compile_time",
            "run_time_mean",
            "run_time_se",
            "errors",
        ]
    )

    for idx, model_name in enumerate(model_list):
        print(f"Processing {model_name} ({idx + 1}/{len(model_list)})")

        model_path = model_name.replace("/", "_")

        t_start = time.time()

        lf = os.environ.get("TORCH_USE_DEFAULT_SCORE_FUSION") != "1"
        score_fusion_type = (
            "default"
            if os.environ.get("TORCH_USE_DEFAULT_SCORE_FUSION") == "1"
            else "random"
            if os.environ.get("TORCH_USE_RANDOM_SCORE_FUSION") == "1"
            else "learned"
        )

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
            tokenizer.pad_token = tokenizer.eos_token
            if not tokenizer.pad_token:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model = (
                model_class.from_pretrained(model_name, trust_remote_code=True)
                .eval()
                .to("cuda")
            )

            # Tokenize sample inputs
            encoded = tokenizer(
                "This is a warmup input.",
                return_tensors="pt",
                padding="max_length",
                max_length=MAX_LEN,
            )
            encoded_dict = {k: v.to("cuda") for k, v in encoded.items()}

            compiled, tc = compile_model(model, encoded_dict, model_path)

            tr = time_model_run(compiled, tokenizer, repeats=10)

            output_df.loc[len(output_df)] = {
                "model": model_name,
                "score_fusion_type": score_fusion_type,
                "compile_time": tc,
                "run_time_mean": np.mean(tr),
                "run_time_se": st.sem(tr),
                "errors": None,
            }

            print(
                f"\t- [SUCCESS] Finished processing {model_name} ({round(time.time() - t_start, 3)}s)"
            )

            torch.cuda.empty_cache()

            output_df.to_csv(
                os.path.join(OUTPUT_DIR, f"evaluation_results_{run_id}.csv"),
                index=False,
            )

            time.sleep(2)

        except Exception as e:
            torch.cuda.empty_cache()

            output_df.loc[len(output_df)] = {
                "model": model_name,
                "score_fusion_type": score_fusion_type,
                "compile_time": None,
                "run_time_mean": None,
                "run_time_se": None,
                "errors": str(e),
            }

            print(
                f"\t- [FAIL] Finished processing {model_name} ({round(time.time() - t_start, 3)}s)"
            )
            print(traceback.format_exc())

            output_df.to_csv(
                os.path.join(OUTPUT_DIR, f"evaluation_results_{run_id}.csv"),
                index=False,
            )

            time.sleep(2)

    failed_models = output_df[output_df["errors"].notna()]
    print(f"Successfully finished {len(output_df) - len(failed_models)} models.")
    print(f"Failed: {failed_models['model'].tolist()}")

    output_df.to_csv(
        os.path.join(OUTPUT_DIR, f"evaluation_results_{run_id}.csv"), index=False
    )


if __name__ == "__main__":
    main()
