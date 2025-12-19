import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
)


# sample script to demonstrate compiling bert-base-uncased in PyTorch.
def main():
    model_name = "bert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    print(f"Loading model {model_name}...")
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare input
    input_text = "I am that merry wanderer of the night."
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compile the model using TorchInductor
    print("Compiling the model with TorchInductor...")
    compiled_model = torch.compile(model, backend="inductor", fullgraph=False)

    # Run inference with the compiled model
    outputs = None
    with torch.no_grad():
        outputs = compiled_model(**inputs)

    print("Model output:", outputs)


if __name__ == "__main__":
    main()
