
import os
from datasets import load_dataset
import pandas as pd

def download_hf_models_dataset(output_csv_path: str):
    ds_name = "breadlicker45/huggingface-models-15M"
    ds_name = "midah/base_models_to_process"
    ds = load_dataset(ds_name, split="train")

    df = pd.DataFrame(ds)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(df)} rows to {output_csv_path}")

if __name__ == "__main__":
    download_hf_models_dataset("data/models.csv")
