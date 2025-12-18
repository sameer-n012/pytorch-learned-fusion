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
